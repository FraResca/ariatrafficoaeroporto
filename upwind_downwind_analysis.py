#!/usr/bin/env python3
"""Analyze airport upwind/downwind regimes for pollutants at Porta San Felice."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
import math
from pathlib import Path
import sys
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import requests
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from analysis_runtime import resolve_workers


DEFAULT_INPUT = Path("Datasets_Raw/hourly_merged_2023_2025.csv")
DEFAULT_OUTPUT_DIR = Path("Analysis")
DEFAULT_SPIRE_INPUT = Path("Datasets_Raw/spire_flow_2023_2025.csv")
DATETIME_COLUMN = "datetime"
POLLUTANTS = [
    "NO2_porta_san_felice",
    "CO_porta_san_felice",
    "C6H6_porta_san_felice",
    "NO2_giardini_margherita",
    "NO2_via_chiarini",
    "O3_giardini_margherita",
    "O3_via_chiarini",
]
DEFAULT_HORIZONS = [0, 1, 3]
DEFAULT_BLQ_LAGS = [0, 1, 2, 3, 6]
AIRPORT_REFERENCE = (44.5354, 11.2887)
# Approximate station coordinates used only for directional wind diagnostics.
# They should be replaced by official ARPAE station metadata for publication-grade work.
STATIONS = {
    "porta_san_felice": {
        "label": "Porta San Felice",
        "lat": 44.5013,
        "lon": 11.3280,
        "pollutants": {
            "NO2": "NO2_porta_san_felice",
            "CO": "CO_porta_san_felice",
            "C6H6": "C6H6_porta_san_felice",
        },
    },
    "giardini_margherita": {
        "label": "Giardini Margherita",
        "lat": 44.4849,
        "lon": 11.3546,
        "pollutants": {"NO2": "NO2_giardini_margherita", "O3": "O3_giardini_margherita"},
    },
    "via_chiarini": {
        "label": "Via Chiarini",
        "lat": 44.4946,
        "lon": 11.3768,
        "pollutants": {"NO2": "NO2_via_chiarini", "O3": "O3_via_chiarini"},
    },
}
SPATIAL_GRADIENTS = {
    "NO2_psf_minus_giardini": ("NO2_porta_san_felice", "NO2_giardini_margherita"),
    "NO2_psf_minus_chiarini": ("NO2_porta_san_felice", "NO2_via_chiarini"),
    "NO2_chiarini_minus_giardini": ("NO2_via_chiarini", "NO2_giardini_margherita"),
    "O3_chiarini_minus_giardini": ("O3_via_chiarini", "O3_giardini_margherita"),
}
PLOT_FORMAT = "svg"
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analisi upwind/downwind per verificare se il segnale aeroportuale "
            "aumenta quando Porta San Felice è sottovento rispetto a BLQ."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", nargs="+", default=POLLUTANTS)
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    parser.add_argument("--blq-activity", default="blq_flights")
    parser.add_argument("--alignment-threshold", type=float, default=0.5)
    parser.add_argument("--min-wind-component", type=float, default=0.5)
    parser.add_argument("--wind-source", choices=["aero", "centro"], default="aero")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--shap-sample", type=int, default=1000)
    parser.add_argument("--blq-lags", nargs="+", type=int, default=DEFAULT_BLQ_LAGS)
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    parser.add_argument("--bootstrap-block-hours", type=int, default=24)
    parser.add_argument("--matching-sample", type=int, default=1200)
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Parallelismo locale. 0=auto: usa SLURM_CPUS_PER_TASK se presente, "
            "altrimenti tutte le CPU locali."
        ),
    )
    return parser.parse_args()


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    df = pd.read_csv(path)
    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], errors="coerce")
    if df[DATETIME_COLUMN].isna().any():
        raise ValueError(f"Date non valide in {DATETIME_COLUMN}")
    return df.sort_values(DATETIME_COLUMN).reset_index(drop=True)


def load_selected_spires(path: Path = DEFAULT_SPIRE_INPUT) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        from merge_hourly_datasets import select_spires
    except Exception:
        return pd.DataFrame()
    df = pd.read_csv(path)
    selected = select_spires(df)
    if selected.empty:
        return selected
    return selected.sort_values(["area", "rank"]).reset_index(drop=True)


def initial_bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    delta_lambda = np.deg2rad(lon2 - lon1)
    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    return float((np.rad2deg(np.arctan2(y, x)) + 360) % 360)


def signed_angle_diff_degrees(direction: pd.Series, target_direction: float) -> pd.Series:
    return (direction - target_direction + 180) % 360 - 180


def add_station_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for station_id, station in STATIONS.items():
        bearing = initial_bearing_degrees(
            AIRPORT_REFERENCE[0],
            AIRPORT_REFERENCE[1],
            station["lat"],
            station["lon"],
        )
        out[f"airport_to_{station_id}_bearing"] = bearing
        for source in ["aero", "centro"]:
            direction_col = f"W_VEC_DIR_{source}"
            intensity_col = f"W_VEC_INT_{source}"
            if direction_col not in out.columns or intensity_col not in out.columns:
                continue
            wind_to_direction = (out[direction_col] + 180) % 360
            angle_diff = signed_angle_diff_degrees(wind_to_direction, bearing)
            alignment = np.cos(np.deg2rad(angle_diff))
            out[f"airport_to_{station_id}_wind_alignment_{source}"] = alignment
            out[f"airport_to_{station_id}_wind_component_{source}"] = out[intensity_col] * alignment
            out[f"airport_to_{station_id}_crosswind_component_{source}"] = out[intensity_col] * np.sin(np.deg2rad(angle_diff))
    return out


def classify_wind_regime(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    alignment_col = f"airport_to_psf_wind_alignment_{args.wind_source}"
    component_col = f"airport_to_psf_wind_component_{args.wind_source}"
    for column in [alignment_col, component_col, args.blq_activity]:
        if column not in out.columns:
            raise ValueError(f"Colonna mancante: {column}")

    alignment = out[alignment_col]
    component = out[component_col]
    moving = component.abs() >= args.min_wind_component
    downwind = moving & (alignment >= args.alignment_threshold) & (component > 0)
    upwind = moving & (alignment <= -args.alignment_threshold) & (component < 0)
    regime = np.select(
        [downwind, upwind, ~moving],
        ["downwind", "upwind", "calm"],
        default="crosswind",
    )
    out["wind_regime"] = regime
    out["downwind_flag"] = (regime == "downwind").astype(int)
    out["upwind_flag"] = (regime == "upwind").astype(int)
    out["crosswind_flag"] = (regime == "crosswind").astype(int)
    out["calm_flag"] = (regime == "calm").astype(int)
    out["blq_activity"] = out[args.blq_activity]
    out["blq_x_downwind"] = out["blq_activity"] * out["downwind_flag"]
    out["blq_x_upwind"] = out["blq_activity"] * out["upwind_flag"]
    return out


def add_multistation_regimes(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    for station_id in STATIONS:
        alignment_col = f"airport_to_{station_id}_wind_alignment_{args.wind_source}"
        component_col = f"airport_to_{station_id}_wind_component_{args.wind_source}"
        if alignment_col not in out.columns or component_col not in out.columns:
            continue
        alignment = out[alignment_col]
        component = out[component_col]
        moving = component.abs() >= args.min_wind_component
        downwind = moving & (alignment >= args.alignment_threshold) & (component > 0)
        upwind = moving & (alignment <= -args.alignment_threshold) & (component < 0)
        regime = np.select([downwind, upwind, ~moving], ["downwind", "upwind", "calm"], default="crosswind")
        out[f"{station_id}_wind_regime"] = regime
        out[f"{station_id}_downwind_flag"] = (regime == "downwind").astype(int)
        out[f"{station_id}_upwind_flag"] = (regime == "upwind").astype(int)
        out[f"blq_x_{station_id}_downwind"] = out["blq_activity"] * out[f"{station_id}_downwind_flag"]
        out[f"blq_x_{station_id}_upwind"] = out["blq_activity"] * out[f"{station_id}_upwind_flag"]
    return out


def add_control_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out[DATETIME_COLUMN]
    hour = dt.dt.hour
    month = dt.dt.month
    day = dt.dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["dayofweek"] = day
    out["is_weekend"] = (day >= 5).astype(int)

    spire_cols = [c for c in out.columns if c.startswith("spire_")]
    airport_spire_cols = [c for c in spire_cols if c.startswith("spire_airport_")]
    urban_cols = [c for c in spire_cols if c not in airport_spire_cols]
    out["urban_traffic_total"] = out[urban_cols].sum(axis=1, min_count=1)
    out["airport_spire_traffic_total"] = out[airport_spire_cols].sum(axis=1, min_count=1)
    out["season"] = np.select(
        [
            month.isin([12, 1, 2]),
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8]),
            month.isin([9, 10, 11]),
        ],
        ["winter", "spring", "summer", "autumn"],
        default="unknown",
    )
    out["night_flag"] = ((hour < 7) | (hour >= 21)).astype(int)
    out["low_ventilation_flag"] = (
        (out[["W_SCAL_INT_aero", "W_SCAL_INT_centro"]].mean(axis=1) <= 1.0)
    ).astype(int)
    return out


def descriptive_summary(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for target in targets:
        for regime, group in df.dropna(subset=[target]).groupby("wind_regime"):
            rows.append(
                {
                    "target": target,
                    "wind_regime": regime,
                    "rows": len(group),
                    "mean": group[target].mean(),
                    "median": group[target].median(),
                    "std": group[target].std(),
                    "p25": group[target].quantile(0.25),
                    "p75": group[target].quantile(0.75),
                    "mean_blq_activity": group["blq_activity"].mean(),
                    "mean_urban_traffic_total": group["urban_traffic_total"].mean(),
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "wind_regime"])


def design_columns(df: pd.DataFrame, target: str, include_target_lag: bool) -> list[str]:
    meteo = [
        "TAVG_aero",
        "PREC_aero",
        "RHAVG_aero",
        "RAD_aero",
        "W_SCAL_INT_aero",
        "W_VEC_INT_aero",
        "TAVG_centro",
        "PREC_centro",
        "RHAVG_centro",
        "RAD_centro",
        "W_SCAL_INT_centro",
        "W_VEC_INT_centro",
    ]
    wind = [
        "airport_to_psf_wind_alignment_aero",
        "airport_to_psf_wind_component_aero",
        "airport_to_psf_crosswind_component_aero",
        "airport_to_psf_wind_alignment_centro",
        "airport_to_psf_wind_component_centro",
        "airport_to_psf_crosswind_component_centro",
    ]
    calendar = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek", "is_weekend"]
    traffic = [
        "blq_activity",
        "downwind_flag",
        "upwind_flag",
        "blq_x_downwind",
        "blq_x_upwind",
        "urban_traffic_total",
        "airport_spire_traffic_total",
    ]
    columns = [c for c in traffic + meteo + wind + calendar if c in df.columns]
    if include_target_lag:
        columns.append(f"{target}_lag_1h")
    return columns


def ols_standardized(data: pd.DataFrame, y_col: str, x_cols: list[str]) -> pd.DataFrame:
    clean = data.dropna(subset=[y_col]).copy()
    if clean.empty:
        return pd.DataFrame()

    x = SimpleImputer(strategy="median").fit_transform(clean[x_cols])
    y = clean[y_col].to_numpy(dtype=float)
    x_scaled = StandardScaler().fit_transform(x)
    y_scaled = (y - y.mean()) / y.std(ddof=0)
    x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])

    beta, _, _, _ = np.linalg.lstsq(x_design, y_scaled, rcond=None)
    pred = x_design @ beta
    resid = y_scaled - pred
    n, p = x_design.shape
    dof = n - p
    sigma2 = float((resid @ resid) / dof)
    xtx_inv = np.linalg.pinv(x_design.T @ x_design)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    t_stat = beta / se
    p_value = 2 * stats.t.sf(np.abs(t_stat), df=dof)
    r2 = 1 - float((resid @ resid) / ((y_scaled - y_scaled.mean()) @ (y_scaled - y_scaled.mean())))

    rows = []
    for name, coef, coef_se, t_val, p_val in zip(["intercept", *x_cols], beta, se, t_stat, p_value):
        rows.append(
            {
                "term": name,
                "std_coef": coef,
                "std_error": coef_se,
                "t_stat": t_val,
                "p_value": p_val,
                "n_rows": n,
                "r2": r2,
            }
        )
    return pd.DataFrame(rows)


def import_shap():
    try:
        import shap as shap_module
    except ImportError:  # pragma: no cover
        return None
    return shap_module


_GLOBAL_DF: pd.DataFrame | None = None
_GLOBAL_ARGS: argparse.Namespace | None = None


def _init_worker(df: pd.DataFrame, args: argparse.Namespace) -> None:
    global _GLOBAL_DF, _GLOBAL_ARGS
    _GLOBAL_DF = df
    _GLOBAL_ARGS = args


def interaction_regressions(df: pd.DataFrame, targets: list[str], horizons: list[int]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for target in targets:
        if target not in df.columns:
            continue
        for horizon in horizons:
            data = df.copy()
            y_col = f"{target}_t_plus_{horizon}h"
            data[y_col] = data[target].shift(-horizon)
            data[f"{target}_lag_1h"] = data[target].shift(1)
            for include_lag in [False, True]:
                feature_set = "with_target_lag_1h" if include_lag else "no_target_lag"
                x_cols = design_columns(data, target, include_lag)
                result = ols_standardized(data, y_col, x_cols)
                if result.empty:
                    continue
                result.insert(0, "target", target)
                result.insert(1, "horizon_h", horizon)
                result.insert(2, "feature_set", feature_set)
                rows.append(result)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _interaction_task(task: tuple[str, int]) -> pd.DataFrame:
    target, horizon = task
    if target not in _GLOBAL_DF.columns:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    data = _GLOBAL_DF.copy()
    y_col = f"{target}_t_plus_{horizon}h"
    data[y_col] = data[target].shift(-horizon)
    data[f"{target}_lag_1h"] = data[target].shift(1)
    for include_lag in [False, True]:
        feature_set = "with_target_lag_1h" if include_lag else "no_target_lag"
        x_cols = design_columns(data, target, include_lag)
        result = ols_standardized(data, y_col, x_cols)
        if result.empty:
            continue
        result.insert(0, "target", target)
        result.insert(1, "horizon_h", horizon)
        result.insert(2, "feature_set", feature_set)
        rows.append(result)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def add_blq_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        col = f"blq_lag_{lag}h"
        out[col] = out["blq_activity"].shift(lag)
        out[f"{col}_x_downwind"] = out[col] * out["downwind_flag"]
        out[f"{col}_x_upwind"] = out[col] * out["upwind_flag"]
    return out


def distributed_lag_regressions(df: pd.DataFrame, targets: list[str], horizons: list[int], blq_lags: list[int]) -> pd.DataFrame:
    lagged = add_blq_lags(df, blq_lags)
    rows: list[pd.DataFrame] = []
    base_controls = [
        "downwind_flag",
        "upwind_flag",
        "urban_traffic_total",
        "airport_spire_traffic_total",
        "TAVG_aero",
        "PREC_aero",
        "RHAVG_aero",
        "RAD_aero",
        "W_SCAL_INT_aero",
        "W_VEC_INT_aero",
        "TAVG_centro",
        "PREC_centro",
        "RHAVG_centro",
        "RAD_centro",
        "W_SCAL_INT_centro",
        "W_VEC_INT_centro",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dayofweek",
        "is_weekend",
    ]
    lag_cols = [
        c
        for lag in blq_lags
        for c in [f"blq_lag_{lag}h", f"blq_lag_{lag}h_x_downwind", f"blq_lag_{lag}h_x_upwind"]
    ]
    for target in targets:
        if target not in lagged.columns:
            continue
        for horizon in horizons:
            data = lagged.copy()
            y_col = f"{target}_t_plus_{horizon}h"
            data[y_col] = data[target].shift(-horizon)
            data[f"{target}_lag_1h"] = data[target].shift(1)
            for include_lag in [False, True]:
                feature_set = "with_target_lag_1h" if include_lag else "no_target_lag"
                cols = [c for c in lag_cols + base_controls if c in data.columns]
                if include_lag:
                    cols.append(f"{target}_lag_1h")
                result = ols_standardized(data, y_col, cols)
                if result.empty:
                    continue
                result.insert(0, "target", target)
                result.insert(1, "horizon_h", horizon)
                result.insert(2, "feature_set", feature_set)
                rows.append(result)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _distributed_lag_task(task: tuple[str, int]) -> pd.DataFrame:
    target, horizon = task
    lagged = add_blq_lags(_GLOBAL_DF, _GLOBAL_ARGS.blq_lags)
    rows: list[pd.DataFrame] = []
    base_controls = [
        "downwind_flag",
        "upwind_flag",
        "urban_traffic_total",
        "airport_spire_traffic_total",
        "TAVG_aero",
        "PREC_aero",
        "RHAVG_aero",
        "RAD_aero",
        "W_SCAL_INT_aero",
        "W_VEC_INT_aero",
        "TAVG_centro",
        "PREC_centro",
        "RHAVG_centro",
        "RAD_centro",
        "W_SCAL_INT_centro",
        "W_VEC_INT_centro",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dayofweek",
        "is_weekend",
    ]
    lag_cols = [
        c
        for lag in _GLOBAL_ARGS.blq_lags
        for c in [f"blq_lag_{lag}h", f"blq_lag_{lag}h_x_downwind", f"blq_lag_{lag}h_x_upwind"]
    ]
    if target not in lagged.columns:
        return pd.DataFrame()
    data = lagged.copy()
    y_col = f"{target}_t_plus_{horizon}h"
    data[y_col] = data[target].shift(-horizon)
    data[f"{target}_lag_1h"] = data[target].shift(1)
    for include_lag in [False, True]:
        feature_set = "with_target_lag_1h" if include_lag else "no_target_lag"
        cols = [c for c in lag_cols + base_controls if c in data.columns]
        if include_lag:
            cols.append(f"{target}_lag_1h")
        result = ols_standardized(data, y_col, cols)
        if result.empty:
            continue
        result.insert(0, "target", target)
        result.insert(1, "horizon_h", horizon)
        result.insert(2, "feature_set", feature_set)
        rows.append(result)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_interaction_summary(regression: pd.DataFrame) -> pd.DataFrame:
    key_terms = [
        "blq_activity",
        "downwind_flag",
        "upwind_flag",
        "blq_x_downwind",
        "blq_x_upwind",
    ]
    summary = regression.loc[regression["term"].isin(key_terms)].copy()
    return summary.sort_values(["target", "horizon_h", "feature_set", "term"])


def quantile_regime_summary(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    out = df.copy()
    q25 = out["blq_activity"].quantile(0.25)
    q75 = out["blq_activity"].quantile(0.75)
    out["blq_activity_class"] = np.select(
        [out["blq_activity"] <= q25, out["blq_activity"] >= q75],
        ["low_blq", "high_blq"],
        default="mid_blq",
    )
    rows: list[dict] = []
    for target in targets:
        if target not in out.columns:
            continue
        for (regime, activity_class), group in out.dropna(subset=[target]).groupby(["wind_regime", "blq_activity_class"]):
            rows.append(
                {
                    "target": target,
                    "wind_regime": regime,
                    "blq_activity_class": activity_class,
                    "rows": len(group),
                    "mean": group[target].mean(),
                    "median": group[target].median(),
                    "mean_blq_activity": group["blq_activity"].mean(),
                    "mean_urban_traffic_total": group["urban_traffic_total"].mean(),
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "wind_regime", "blq_activity_class"])


def matched_downwind_upwind(df: pd.DataFrame, targets: list[str], sample_limit: int) -> pd.DataFrame:
    match_cols = [
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dayofweek",
        "is_weekend",
        "TAVG_aero",
        "W_SCAL_INT_aero",
        "W_VEC_INT_aero",
        "urban_traffic_total",
        "blq_activity",
    ]
    needed = match_cols + targets
    data = df.loc[df["wind_regime"].isin(["downwind", "upwind"])].dropna(subset=needed).copy()
    down = data.loc[data["wind_regime"] == "downwind"].copy()
    up = data.loc[data["wind_regime"] == "upwind"].copy()
    if down.empty or up.empty:
        return pd.DataFrame()
    if len(down) > sample_limit:
        down = down.sample(n=sample_limit, random_state=42)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    up_x = scaler.fit_transform(imputer.fit_transform(up[match_cols]))
    down_x = scaler.transform(imputer.transform(down[match_cols]))
    rows: list[dict] = []
    used_up: set[int] = set()
    up_indices = up.index.to_numpy()
    for i, down_idx in enumerate(down.index):
        distances = ((up_x - down_x[i]) ** 2).sum(axis=1)
        order = np.argsort(distances)
        chosen_pos = None
        for pos in order:
            if int(up_indices[pos]) not in used_up:
                chosen_pos = pos
                used_up.add(int(up_indices[pos]))
                break
        if chosen_pos is None:
            break
        up_idx = int(up_indices[chosen_pos])
        row = {
            "downwind_datetime": down.loc[down_idx, DATETIME_COLUMN],
            "upwind_datetime": up.loc[up_idx, DATETIME_COLUMN],
            "distance": float(np.sqrt(distances[chosen_pos])),
            "downwind_blq_activity": down.loc[down_idx, "blq_activity"],
            "upwind_blq_activity": up.loc[up_idx, "blq_activity"],
            "downwind_urban_traffic_total": down.loc[down_idx, "urban_traffic_total"],
            "upwind_urban_traffic_total": up.loc[up_idx, "urban_traffic_total"],
        }
        for target in targets:
            row[f"{target}_downwind"] = down.loc[down_idx, target]
            row[f"{target}_upwind"] = up.loc[up_idx, target]
            row[f"{target}_diff_downwind_minus_upwind"] = down.loc[down_idx, target] - up.loc[up_idx, target]
        rows.append(row)
    return pd.DataFrame(rows)


def matched_summary(matches: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    if matches.empty:
        return pd.DataFrame()
    for target in targets:
        diff = matches[f"{target}_diff_downwind_minus_upwind"].dropna()
        if diff.empty:
            continue
        t_stat, p_value = stats.ttest_1samp(diff, popmean=0.0)
        rows.append(
            {
                "target": target,
                "pairs": len(diff),
                "mean_diff_downwind_minus_upwind": diff.mean(),
                "median_diff_downwind_minus_upwind": diff.median(),
                "std_diff": diff.std(),
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def daily_block_bootstrap_effects(df: pd.DataFrame, targets: list[str], repeats: int, block_hours: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    out = df.copy()
    out["block"] = np.arange(len(out)) // block_hours
    blocks = out["block"].unique()
    rows: list[dict] = []
    for target in targets:
        clean = out.dropna(subset=[target]).copy()
        for contrast in ["downwind_minus_upwind", "high_downwind_minus_low_downwind"]:
            values: list[float] = []
            for _ in range(repeats):
                sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
                sample = pd.concat([clean.loc[clean["block"] == block] for block in sampled_blocks], ignore_index=True)
                if contrast == "downwind_minus_upwind":
                    a = sample.loc[sample["wind_regime"] == "downwind", target]
                    b = sample.loc[sample["wind_regime"] == "upwind", target]
                else:
                    q25 = clean["blq_activity"].quantile(0.25)
                    q75 = clean["blq_activity"].quantile(0.75)
                    down = sample.loc[sample["wind_regime"] == "downwind"]
                    a = down.loc[down["blq_activity"] >= q75, target]
                    b = down.loc[down["blq_activity"] <= q25, target]
                if len(a) and len(b):
                    values.append(float(a.mean() - b.mean()))
            if values:
                rows.append(
                    {
                        "target": target,
                        "contrast": contrast,
                        "repeats": len(values),
                        "mean_effect": float(np.mean(values)),
                        "ci95_low": float(np.quantile(values, 0.025)),
                        "ci95_high": float(np.quantile(values, 0.975)),
                    }
                )
    return pd.DataFrame(rows)


def threshold_sensitivity(df: pd.DataFrame, targets: list[str], min_wind_component: float, wind_source: str) -> pd.DataFrame:
    rows: list[dict] = []
    base_args = argparse.Namespace(wind_source=wind_source, min_wind_component=min_wind_component, blq_activity="blq_activity")
    for threshold in [0.3, 0.5, 0.7, 0.85]:
        args = argparse.Namespace(**vars(base_args), alignment_threshold=threshold)
        classified = classify_wind_regime(df.drop(columns=[c for c in ["wind_regime", "downwind_flag", "upwind_flag", "crosswind_flag", "calm_flag", "blq_x_downwind", "blq_x_upwind"] if c in df.columns]), args)
        counts = classified["wind_regime"].value_counts()
        for target in targets:
            down = classified.loc[classified["wind_regime"] == "downwind", target].dropna()
            up = classified.loc[classified["wind_regime"] == "upwind", target].dropna()
            rows.append(
                {
                    "target": target,
                    "alignment_threshold": threshold,
                    "downwind_rows": int(counts.get("downwind", 0)),
                    "upwind_rows": int(counts.get("upwind", 0)),
                    "downwind_mean": down.mean(),
                    "upwind_mean": up.mean(),
                    "downwind_minus_upwind": down.mean() - up.mean(),
                }
            )
    return pd.DataFrame(rows)


def add_spatial_gradients(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for gradient, (left, right) in SPATIAL_GRADIENTS.items():
        if left in out.columns and right in out.columns:
            out[gradient] = out[left] - out[right]
    return out


def spatial_gradient_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for gradient in SPATIAL_GRADIENTS:
        if gradient not in df.columns:
            continue
        for regime, group in df.dropna(subset=[gradient]).groupby("wind_regime"):
            rows.append(
                {
                    "gradient": gradient,
                    "wind_regime": regime,
                    "rows": len(group),
                    "mean": group[gradient].mean(),
                    "median": group[gradient].median(),
                    "std": group[gradient].std(),
                    "mean_blq_activity": group["blq_activity"].mean(),
                    "mean_urban_traffic_total": group["urban_traffic_total"].mean(),
                }
            )
    return pd.DataFrame(rows).sort_values(["gradient", "wind_regime"])


def did_design_columns(df: pd.DataFrame, gradient: str) -> list[str]:
    meteo = [
        "TAVG_aero",
        "PREC_aero",
        "RHAVG_aero",
        "RAD_aero",
        "W_SCAL_INT_aero",
        "W_VEC_INT_aero",
        "TAVG_centro",
        "PREC_centro",
        "RHAVG_centro",
        "RAD_centro",
        "W_SCAL_INT_centro",
        "W_VEC_INT_centro",
    ]
    calendar = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek", "is_weekend"]
    traffic = ["blq_activity", "urban_traffic_total", "airport_spire_traffic_total"]
    wind_controls = [
        c
        for c in df.columns
        if c.startswith("airport_to_")
        and (
            c.endswith(f"_wind_alignment_aero")
            or c.endswith(f"_wind_component_aero")
            or c.endswith(f"_crosswind_component_aero")
        )
    ]
    station_terms: list[str] = []
    if "psf_minus_giardini" in gradient:
        station_terms = [
            "porta_san_felice_downwind_flag",
            "giardini_margherita_downwind_flag",
            "blq_x_porta_san_felice_downwind",
            "blq_x_giardini_margherita_downwind",
        ]
    elif "psf_minus_chiarini" in gradient:
        station_terms = [
            "porta_san_felice_downwind_flag",
            "via_chiarini_downwind_flag",
            "blq_x_porta_san_felice_downwind",
            "blq_x_via_chiarini_downwind",
        ]
    elif "chiarini_minus_giardini" in gradient:
        station_terms = [
            "via_chiarini_downwind_flag",
            "giardini_margherita_downwind_flag",
            "blq_x_via_chiarini_downwind",
            "blq_x_giardini_margherita_downwind",
        ]
    return [c for c in traffic + station_terms + meteo + wind_controls + calendar if c in df.columns]


def did_regressions(df: pd.DataFrame, gradients: list[str], horizons: list[int]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for gradient in gradients:
        if gradient not in df.columns:
            continue
        for horizon in horizons:
            data = df.copy()
            y_col = f"{gradient}_t_plus_{horizon}h"
            data[y_col] = data[gradient].shift(-horizon)
            data[f"{gradient}_lag_1h"] = data[gradient].shift(1)
            for include_lag in [False, True]:
                feature_set = "with_gradient_lag_1h" if include_lag else "no_gradient_lag"
                cols = did_design_columns(data, gradient)
                if include_lag:
                    cols.append(f"{gradient}_lag_1h")
                result = ols_standardized(data, y_col, cols)
                if result.empty:
                    continue
                result.insert(0, "gradient", gradient)
                result.insert(1, "horizon_h", horizon)
                result.insert(2, "feature_set", feature_set)
                rows.append(result)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _did_task(task: tuple[str, int]) -> pd.DataFrame:
    gradient, horizon = task
    if gradient not in _GLOBAL_DF.columns:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    data = _GLOBAL_DF.copy()
    y_col = f"{gradient}_t_plus_{horizon}h"
    data[y_col] = data[gradient].shift(-horizon)
    data[f"{gradient}_lag_1h"] = data[gradient].shift(1)
    for include_lag in [False, True]:
        feature_set = "with_gradient_lag_1h" if include_lag else "no_gradient_lag"
        cols = did_design_columns(data, gradient)
        if include_lag:
            cols.append(f"{gradient}_lag_1h")
        result = ols_standardized(data, y_col, cols)
        if result.empty:
            continue
        result.insert(0, "gradient", gradient)
        result.insert(1, "horizon_h", horizon)
        result.insert(2, "feature_set", feature_set)
        rows.append(result)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def did_summary(did: pd.DataFrame) -> pd.DataFrame:
    if did.empty:
        return pd.DataFrame()
    terms = did.loc[did["term"].str.startswith("blq_x_") & did["term"].str.endswith("_downwind")].copy()
    terms["station"] = terms["term"].str.removeprefix("blq_x_").str.removesuffix("_downwind")
    return terms.sort_values(["gradient", "horizon_h", "feature_set", "station"])


def station_panel(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for station_id, station in STATIONS.items():
        for pollutant, column in station["pollutants"].items():
            if column not in df.columns:
                continue
            station_df = pd.DataFrame(
                {
                    DATETIME_COLUMN: df[DATETIME_COLUMN],
                    "station": station_id,
                    "station_label": station["label"],
                    "pollutant": pollutant,
                    "value": df[column],
                    "blq_activity": df["blq_activity"],
                    "urban_traffic_total": df["urban_traffic_total"],
                    "downwind_flag": df.get(f"{station_id}_downwind_flag"),
                    "upwind_flag": df.get(f"{station_id}_upwind_flag"),
                    "blq_x_station_downwind": df.get(f"blq_x_{station_id}_downwind"),
                    "wind_alignment": df.get(f"airport_to_{station_id}_wind_alignment_aero"),
                    "wind_component": df.get(f"airport_to_{station_id}_wind_component_aero"),
                }
            )
            rows.append(station_df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values([DATETIME_COLUMN, "station", "pollutant"])


def temporal_train_test(data: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = int(len(data) * (1 - test_size))
    if split <= 0 or split >= len(data):
        raise ValueError("Split temporale non valido")
    return data.iloc[:split].copy(), data.iloc[split:].copy()


def shap_feature_groups(features: list[str], target: str) -> dict[str, list[str]]:
    return {
        "airport": [
            f
            for f in features
            if f in {"blq_activity", "airport_spire_traffic_total", "blq_x_downwind", "blq_x_upwind"}
        ],
        "urban_traffic": [f for f in features if f == "urban_traffic_total"],
        "wind_regime": [f for f in features if f in {"downwind_flag", "upwind_flag"}],
        "wind_transport": [f for f in features if f.startswith("airport_to_psf_")],
        "meteo": [
            f
            for f in features
            if f.endswith("_aero")
            or f.endswith("_centro")
            or f in {"TAVG_aero", "PREC_aero", "RHAVG_aero", "RAD_aero", "TAVG_centro", "PREC_centro", "RHAVG_centro", "RAD_centro"}
        ],
        "time": [f for f in features if f in {"hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek", "is_weekend"}],
        "target_lag": [f for f in features if f == f"{target}_lag_1h"],
    }


def row_level_shap_by_regime(df: pd.DataFrame, targets: list[str], horizons: list[int], args: argparse.Namespace) -> pd.DataFrame:
    if args.no_shap:
        return pd.DataFrame()
    shap_module = import_shap()
    if shap_module is None:
        return pd.DataFrame()

    rows: list[dict] = []
    for target in targets:
        if target not in df.columns:
            continue
        for horizon in horizons:
            data = df.copy()
            y_col = f"{target}_t_plus_{horizon}h"
            data[y_col] = data[target].shift(-horizon)
            data[f"{target}_lag_1h"] = data[target].shift(1)
            features = design_columns(data, target, include_target_lag=True)
            data = data.dropna(subset=[y_col]).reset_index(drop=True)
            train, test = temporal_train_test(data, args.test_size)

            imputer = SimpleImputer(strategy="median")
            x_train = imputer.fit_transform(train[features])
            x_test = imputer.transform(test[features])
            model = XGBRegressor(
                random_state=42,
                n_estimators=args.n_estimators,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                n_jobs=-1,
            )
            model.fit(x_train, train[y_col])

            sample_size = min(args.shap_sample, len(test))
            sample = test.sample(n=sample_size, random_state=42)
            sample_index = sample.index
            x_sample = imputer.transform(sample[features])
            values = shap_module.TreeExplainer(model).shap_values(x_sample)
            abs_values = np.abs(values)
            groups = shap_feature_groups(features, target)
            feature_index = {feature: i for i, feature in enumerate(features)}

            sample_regimes = test.loc[sample_index, "wind_regime"].to_numpy()
            for regime in sorted(pd.unique(sample_regimes)):
                regime_mask = sample_regimes == regime
                if not regime_mask.any():
                    continue
                for group_name, group_features in groups.items():
                    present = [f for f in group_features if f in feature_index]
                    if not present:
                        continue
                    idx = [feature_index[f] for f in present]
                    rows.append(
                        {
                            "target": target,
                            "horizon_h": horizon,
                            "wind_regime": regime,
                            "group": group_name,
                            "n_rows": int(regime_mask.sum()),
                            "n_features": len(present),
                            "mean_abs_shap": float(abs_values[regime_mask][:, idx].sum(axis=1).mean()),
                        }
                    )
    return pd.DataFrame(rows).sort_values(["target", "horizon_h", "wind_regime", "mean_abs_shap"], ascending=[True, True, True, False])


def save_regime_mean_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    regime_order = ["downwind", "upwind", "crosswind", "calm"]
    for target in sorted(summary["target"].unique()):
        target_df = summary.loc[summary["target"] == target].copy()
        if target_df.empty:
            continue
        target_df["wind_regime"] = pd.Categorical(target_df["wind_regime"], categories=regime_order, ordered=True)
        target_df = target_df.sort_values("wind_regime")
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.bar(target_df["wind_regime"].astype(str), target_df["mean"], color=["#d62728", "#1f77b4", "#7f7f7f", "#bcbd22"][: len(target_df)])
        ax.set_title(f"{target} - mean concentration by wind regime")
        ax.set_xlabel("Wind regime")
        ax.set_ylabel("Mean concentration")
        ax.grid(True, axis="y", alpha=0.3)
        fig.savefig(plots_dir / f"{target}_regime_means.{PLOT_FORMAT}", bbox_inches="tight", format=PLOT_FORMAT)
        plt.close(fig)


def save_threshold_sensitivity_plots(sensitivity: pd.DataFrame, output_dir: Path) -> None:
    if sensitivity.empty:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for target in sorted(sensitivity["target"].unique()):
        target_df = sensitivity.loc[sensitivity["target"] == target].sort_values("alignment_threshold")
        if target_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.plot(
            target_df["alignment_threshold"],
            target_df["downwind_minus_upwind"],
            marker="o",
            linewidth=2,
            color="#1f77b4",
        )
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.8)
        ax.set_title(f"{target} - alignment-threshold sensitivity")
        ax.set_xlabel("Alignment threshold")
        ax.set_ylabel("downwind - upwind")
        ax.grid(True, alpha=0.3)
        fig.savefig(plots_dir / f"{target}_threshold_sensitivity.{PLOT_FORMAT}", bbox_inches="tight", format=PLOT_FORMAT)
        plt.close(fig)


def save_bootstrap_effect_plots(bootstrap: pd.DataFrame, output_dir: Path) -> None:
    if bootstrap.empty:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for contrast in sorted(bootstrap["contrast"].unique()):
        contrast_df = bootstrap.loc[bootstrap["contrast"] == contrast].copy()
        if contrast_df.empty:
            continue
        contrast_df = contrast_df.sort_values("mean_effect")
        y_pos = np.arange(len(contrast_df))
        fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(contrast_df))), constrained_layout=True)
        mean = contrast_df["mean_effect"].to_numpy()
        lower = mean - contrast_df["ci95_low"].to_numpy()
        upper = contrast_df["ci95_high"].to_numpy() - mean
        ax.errorbar(mean, y_pos, xerr=[lower, upper], fmt="o", color="#1f77b4", ecolor="#1f77b4", capsize=4)
        ax.axvline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(contrast_df["target"])
        ax.set_xlabel("confidence interval")
        ax.grid(True, axis="x", alpha=0.3)
        fig.savefig(plots_dir / f"bootstrap_{contrast}.{PLOT_FORMAT}", bbox_inches="tight", format=PLOT_FORMAT)
        plt.close(fig)


def save_shap_regime_plots(shap_context: pd.DataFrame, output_dir: Path) -> None:
    if shap_context.empty:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for target in sorted(shap_context["target"].unique()):
        target_df = shap_context.loc[shap_context["target"] == target].copy()
        if target_df.empty:
            continue
        target_df = (
            target_df.groupby(["wind_regime", "group"], as_index=False)["mean_abs_shap"]
            .mean()
            .sort_values(["wind_regime", "mean_abs_shap"], ascending=[True, False])
        )
        top_groups = (
            target_df.groupby("group", as_index=False)["mean_abs_shap"]
            .mean()
            .sort_values("mean_abs_shap", ascending=False)
            .head(6)["group"]
            .tolist()
        )
        target_df = target_df.loc[target_df["group"].isin(top_groups)]
        if target_df.empty:
            continue
        pivot = target_df.pivot(index="group", columns="wind_regime", values="mean_abs_shap").fillna(0.0)
        pivot = pivot.loc[top_groups]
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"{target} - mean SHAP by feature group and regime")
        ax.set_xlabel("Feature group")
        ax.set_ylabel("mean |SHAP|")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(title="Wind regime", fontsize=8)
        fig.savefig(plots_dir / f"{target}_shap_by_regime.{PLOT_FORMAT}", bbox_inches="tight", format=PLOT_FORMAT)
        plt.close(fig)


def target_station_id(target: str) -> str | None:
    for station_id in STATIONS:
        if station_id in target:
            return station_id
    return None


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile_to_lonlat(x: int, y: int, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def _read_or_download_tile(cache_dir: Path, zoom: int, x: int, y: int) -> Image.Image | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tile_path = cache_dir / f"{zoom}_{x}_{y}.png"
    if tile_path.exists():
        return Image.open(tile_path).convert("RGB")
    url = OSM_TILE_URL.format(z=zoom, x=x, y=y)
    try:
        response = requests.get(
            url,
            timeout=12,
            headers={"User-Agent": "ariatrafficoaeroporto-paper-map/1.0"},
        )
        response.raise_for_status()
        tile = Image.open(BytesIO(response.content)).convert("RGB")
        tile.save(tile_path)
        return tile
    except Exception as exc:
        print(f"Warning: cannot load OSM tile {zoom}/{x}/{y}: {exc}", file=sys.stderr)
        return None


def add_osm_basemap(
    ax: plt.Axes,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    cache_dir: Path,
    zoom: int = 13,
) -> None:
    """Draw an OpenStreetMap basemap while preserving the lon/lat extent."""
    x_min, y_max = _lonlat_to_tile(lon_min, lat_min, zoom)
    x_max, y_min = _lonlat_to_tile(lon_max, lat_max, zoom)
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    rows: list[Image.Image] = []
    for y in range(y_min, y_max + 1):
        row_tiles: list[Image.Image] = []
        for x in range(x_min, x_max + 1):
            tile = _read_or_download_tile(cache_dir, zoom, x, y)
            if tile is None:
                return
            row_tiles.append(tile)
        rows.append(Image.fromarray(np.hstack([np.asarray(tile) for tile in row_tiles])))

    mosaic = Image.fromarray(np.vstack([np.asarray(row) for row in rows]))
    west, north = _tile_to_lonlat(x_min, y_min, zoom)
    east, south = _tile_to_lonlat(x_max + 1, y_max + 1, zoom)
    ax.imshow(
        mosaic,
        extent=(west, east, south, north),
        origin="upper",
        zorder=0,
        alpha=0.58,
        interpolation="bilinear",
    )
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.text(
        0.01,
        0.01,
        "Map data (C) OpenStreetMap contributors",
        transform=ax.transAxes,
        fontsize=6,
        color="#4d4d4d",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
        zorder=20,
    )


def save_geometry_map(output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    airport_lon, airport_lat = AIRPORT_REFERENCE[1], AIRPORT_REFERENCE[0]

    all_lons = [airport_lon, *[station["lon"] for station in STATIONS.values()]]
    all_lats = [airport_lat, *[station["lat"] for station in STATIONS.values()]]
    spires = load_selected_spires()
    if not spires.empty:
        all_lons.extend(spires["longitudine"].dropna().astype(float).tolist())
        all_lats.extend(spires["latitudine"].dropna().astype(float).tolist())
    lon_min, lon_max = min(all_lons), max(all_lons)
    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_pad = (lon_max - lon_min) * 0.05
    lat_pad = (lat_max - lat_min) * 0.05
    lon_min, lon_max = lon_min - lon_pad, lon_max + lon_pad
    lat_min, lat_max = lat_min - lat_pad, lat_max + lat_pad
    add_osm_basemap(ax, lon_min, lon_max, lat_min, lat_max, plots_dir / "_osm_tile_cache")

    ax.scatter(
        [airport_lon],
        [airport_lat],
        s=180,
        marker="*",
        color="#d62728",
        edgecolor="black",
        linewidth=0.5,
        label="BLQ",
        zorder=10,
    )
    ax.text(airport_lon + 0.002, airport_lat - 0.0002, "BLQ", fontsize=10, weight="bold", zorder=11)

    if not spires.empty:
        area_styles = {
            "airport": {"color": "#ff7f0e", "marker": "s", "label": "Airport-side loops"},
            "giardini_margherita": {"color": "#2ca02c", "marker": "^", "label": "Giardini Margherita loops"},
            "porta_san_felice": {"color": "#9467bd", "marker": "D", "label": "Porta San Felice loops"},
            "via_chiarini": {"color": "#8c564b", "marker": "P", "label": "Via Chiarini loops"},
        }
        for area, area_df in spires.groupby("area", sort=False):
            style = area_styles.get(area, {"color": "#7f7f7f", "marker": "o", "label": f"{area} loops"})
            ax.scatter(
                area_df["longitudine"],
                area_df["latitudine"],
                s=50,
                marker=style["marker"],
                color=style["color"],
                edgecolor="black",
                linewidth=0.4,
                alpha=0.85,
                label=style["label"],
                zorder=10,
            )
            for _, row in area_df.iterrows():
                ax.text(
                    row["longitudine"] + 0.0008,
                    row["latitudine"] - 0.0005,
                    f"{int(row['id_uni'])}",
                    fontsize=7,
                    color=style["color"],
                    zorder=11,
                )

    for station_id, station in STATIONS.items():
        lon, lat = station["lon"], station["lat"]
        ax.scatter([lon], [lat], s=90, color="#1f77b4", edgecolor="black", linewidth=0.5, zorder=10)
        ax.text(lon + 0.0015, lat + 0.0010, station["label"], fontsize=9, zorder=11)
        ax.annotate(
            "",
            xy=(lon, lat),
            xytext=(airport_lon, airport_lat),
            arrowprops=dict(arrowstyle="->", color="#4d4d4d", linewidth=1.2, alpha=0.9),
            zorder=9,
        )

    ax.set_title("BLQ, monitoring stations and selected traffic loops")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25, color="white", linewidth=0.8)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1.0 / math.cos(math.radians((lat_min + lat_max) / 2.0)), adjustable="box")
    ax.legend(loc="best", framealpha=0.9)
    fig.savefig(plots_dir / f"airport_station_geometry_map.{PLOT_FORMAT}", bbox_inches="tight", format=PLOT_FORMAT)
    plt.close(fig)


def save_regime_effect_map(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    pivot = (
        summary.loc[summary["wind_regime"].isin(["downwind", "upwind"]), ["target", "wind_regime", "mean"]]
        .pivot(index="target", columns="wind_regime", values="mean")
        .dropna(subset=["downwind", "upwind"], how="all")
    )
    if pivot.empty:
        return
    pivot["downwind_minus_upwind"] = pivot["downwind"] - pivot["upwind"]
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    airport_lon, airport_lat = AIRPORT_REFERENCE[1], AIRPORT_REFERENCE[0]
    ax.scatter([airport_lon], [airport_lat], s=180, marker="*", color="#d62728", label="BLQ")

    all_effects = pivot["downwind_minus_upwind"].to_numpy(dtype=float)
    vmax = max(1e-9, np.nanmax(np.abs(all_effects)))
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    for target, row in pivot.iterrows():
        station_id = target_station_id(target)
        if station_id is None:
            continue
        station = STATIONS[station_id]
        lon, lat = station["lon"], station["lat"]
        effect = float(row["downwind_minus_upwind"])
        ax.scatter([lon], [lat], s=140, color=cmap(norm(effect)), edgecolor="black", linewidth=0.6)
        ax.annotate(
            f"{target}\nΔ={effect:+.2f}",
            xy=(lon, lat),
            xytext=(lon + 0.0018, lat + 0.0010),
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
        )
        ax.annotate(
            "",
            xy=(lon, lat),
            xytext=(airport_lon, airport_lat),
            arrowprops=dict(arrowstyle="->", color="#7f7f7f", linewidth=1.0, alpha=0.6),
        )

    ax.set_title("Descriptive downwind - upwind contrast by target")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("downwind - upwind")
    fig.savefig(plots_dir / f"downwind_upwind_effect_map.{PLOT_FORMAT}", bbox_inches="tight", format=PLOT_FORMAT)
    plt.close(fig)


def _shap_task(task: tuple[str, int]) -> pd.DataFrame:
    target, horizon = task
    if _GLOBAL_ARGS.no_shap or target not in _GLOBAL_DF.columns:
        return pd.DataFrame()
    shap_module = import_shap()
    if shap_module is None:
        return pd.DataFrame()

    data = _GLOBAL_DF.copy()
    y_col = f"{target}_t_plus_{horizon}h"
    data[y_col] = data[target].shift(-horizon)
    data[f"{target}_lag_1h"] = data[target].shift(1)
    features = design_columns(data, target, include_target_lag=True)
    data = data.dropna(subset=[y_col]).reset_index(drop=True)
    train, test = temporal_train_test(data, _GLOBAL_ARGS.test_size)

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    model = XGBRegressor(
        random_state=42,
        n_estimators=_GLOBAL_ARGS.n_estimators,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=1,
    )
    model.fit(x_train, train[y_col])

    sample_size = min(_GLOBAL_ARGS.shap_sample, len(test))
    sample = test.sample(n=sample_size, random_state=42)
    sample_index = sample.index
    x_sample = imputer.transform(sample[features])
    values = shap_module.TreeExplainer(model).shap_values(x_sample)
    abs_values = np.abs(values)
    groups = shap_feature_groups(features, target)
    feature_index = {feature: i for i, feature in enumerate(features)}

    rows: list[dict] = []
    sample_regimes = test.loc[sample_index, "wind_regime"].to_numpy()
    for regime in sorted(pd.unique(sample_regimes)):
        regime_mask = sample_regimes == regime
        if not regime_mask.any():
            continue
        for group_name, group_features in groups.items():
            present = [f for f in group_features if f in feature_index]
            if not present:
                continue
            idx = [feature_index[f] for f in present]
            rows.append(
                {
                    "target": target,
                    "horizon_h": horizon,
                    "wind_regime": regime,
                    "group": group_name,
                    "n_rows": int(regime_mask.sum()),
                    "n_features": len(present),
                    "mean_abs_shap": float(abs_values[regime_mask][:, idx].sum(axis=1).mean()),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def run_parallel(tasks: list, worker_fn, df: pd.DataFrame, args: argparse.Namespace) -> list[pd.DataFrame]:
    if not tasks:
        return []
    workers = resolve_workers(args.workers, len(tasks))
    print(f"Worker {worker_fn.__name__}: {workers}")
    if workers == 1:
        _init_worker(df, args)
        return [worker_fn(task) for task in tasks]
    worker_args = argparse.Namespace(**vars(args))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(df, worker_args),
    ) as executor:
        futures = [executor.submit(worker_fn, task) for task in tasks]
        return [future.result() for future in as_completed(futures)]


def main() -> int:
    args = parse_args()
    df = read_dataset(args.input)
    df = add_station_wind_features(df)
    df = classify_wind_regime(df, args)
    df = add_multistation_regimes(df, args)
    df = add_control_features(df)
    df = add_spatial_gradients(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    classified_path = args.output_dir / "upwind_downwind_classified_hours.csv"
    summary_path = args.output_dir / "upwind_downwind_summary.csv"
    regression_path = args.output_dir / "upwind_downwind_regression_coefficients.csv"
    effect_path = args.output_dir / "upwind_downwind_blq_effects.csv"
    shap_path = args.output_dir / "upwind_downwind_group_shap_by_regime.csv"
    quantile_path = args.output_dir / "upwind_downwind_blq_quantile_summary.csv"
    distributed_lag_path = args.output_dir / "upwind_downwind_distributed_lag_coefficients.csv"
    distributed_lag_effect_path = args.output_dir / "upwind_downwind_distributed_lag_effects.csv"
    matches_path = args.output_dir / "upwind_downwind_matched_pairs.csv"
    matched_summary_path = args.output_dir / "upwind_downwind_matched_summary.csv"
    bootstrap_path = args.output_dir / "upwind_downwind_bootstrap_effects.csv"
    sensitivity_path = args.output_dir / "upwind_downwind_threshold_sensitivity.csv"
    station_wind_path = args.output_dir / "multistation_station_wind_features.csv"
    spatial_gradient_path = args.output_dir / "multistation_spatial_gradients.csv"
    spatial_gradient_summary_path = args.output_dir / "multistation_spatial_gradient_summary.csv"
    did_regression_path = args.output_dir / "multistation_did_regression_coefficients.csv"
    did_summary_path = args.output_dir / "multistation_did_summary.csv"
    panel_path = args.output_dir / "multistation_panel_long.csv"
    plots_dir = args.output_dir / "plots"

    cols_to_save = [
        DATETIME_COLUMN,
        "wind_regime",
        "downwind_flag",
        "upwind_flag",
        "crosswind_flag",
        "calm_flag",
        "blq_activity",
        "blq_x_downwind",
        "blq_x_upwind",
        "urban_traffic_total",
        "airport_spire_traffic_total",
        *[f"{station_id}_wind_regime" for station_id in STATIONS if f"{station_id}_wind_regime" in df.columns],
        *[t for t in args.targets if t in df.columns],
    ]
    df[cols_to_save].to_csv(classified_path, index=False)
    station_wind_cols = [
        DATETIME_COLUMN,
        *[
            column
            for station_id in STATIONS
            for column in [
                f"airport_to_{station_id}_bearing",
                f"airport_to_{station_id}_wind_alignment_aero",
                f"airport_to_{station_id}_wind_component_aero",
                f"airport_to_{station_id}_crosswind_component_aero",
                f"{station_id}_wind_regime",
                f"{station_id}_downwind_flag",
                f"blq_x_{station_id}_downwind",
            ]
            if column in df.columns
        ],
    ]
    df[station_wind_cols].to_csv(station_wind_path, index=False)

    summary = descriptive_summary(df, args.targets)
    regression_tasks = [(target, horizon) for target in args.targets for horizon in args.horizons]
    regression_parts = run_parallel(regression_tasks, _interaction_task, df, args)
    regression = pd.concat([part for part in regression_parts if not part.empty], ignore_index=True) if regression_parts else pd.DataFrame()
    effects = build_interaction_summary(regression)
    quantiles = quantile_regime_summary(df, args.targets)
    distributed_lag_parts = run_parallel(regression_tasks, _distributed_lag_task, df, args)
    distributed_lag = (
        pd.concat([part for part in distributed_lag_parts if not part.empty], ignore_index=True)
        if distributed_lag_parts
        else pd.DataFrame()
    )
    if distributed_lag.empty:
        distributed_lag_effects = pd.DataFrame()
    else:
        distributed_lag_effects = distributed_lag.loc[
            distributed_lag["term"].str.contains("blq_lag_") & distributed_lag["term"].str.contains("_x_downwind|_x_upwind")
        ].copy()
    matches = matched_downwind_upwind(df, args.targets, args.matching_sample)
    match_summary = matched_summary(matches, args.targets)
    bootstrap = daily_block_bootstrap_effects(df, args.targets, args.bootstrap_repeats, args.bootstrap_block_hours)
    sensitivity = threshold_sensitivity(df, args.targets, args.min_wind_component, args.wind_source)
    gradient_cols = [gradient for gradient in SPATIAL_GRADIENTS if gradient in df.columns]
    spatial_gradients = df[[DATETIME_COLUMN, "wind_regime", "blq_activity", "urban_traffic_total", *gradient_cols]].copy()
    spatial_summary = spatial_gradient_summary(df)
    did_tasks = [(gradient, horizon) for gradient in gradient_cols for horizon in args.horizons]
    did_parts = run_parallel(did_tasks, _did_task, df, args)
    did = pd.concat([part for part in did_parts if not part.empty], ignore_index=True) if did_parts else pd.DataFrame()
    did_key = did_summary(did)
    panel = station_panel(df)
    shap_parts = run_parallel(regression_tasks, _shap_task, df, args)
    shap_context = (
        pd.concat([part for part in shap_parts if not part.empty], ignore_index=True)
        .sort_values(["target", "horizon_h", "wind_regime", "mean_abs_shap"], ascending=[True, True, True, False])
        if shap_parts
        else pd.DataFrame()
    )

    summary.to_csv(summary_path, index=False)
    regression.to_csv(regression_path, index=False)
    effects.to_csv(effect_path, index=False)
    quantiles.to_csv(quantile_path, index=False)
    distributed_lag.to_csv(distributed_lag_path, index=False)
    distributed_lag_effects.to_csv(distributed_lag_effect_path, index=False)
    matches.to_csv(matches_path, index=False)
    match_summary.to_csv(matched_summary_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    spatial_gradients.to_csv(spatial_gradient_path, index=False)
    spatial_summary.to_csv(spatial_gradient_summary_path, index=False)
    did.to_csv(did_regression_path, index=False)
    did_key.to_csv(did_summary_path, index=False)
    panel.to_csv(panel_path, index=False)
    if not shap_context.empty:
        shap_context.to_csv(shap_path, index=False)

    save_regime_mean_plots(summary, args.output_dir)
    save_geometry_map(args.output_dir)
    save_regime_effect_map(summary, args.output_dir)
    save_threshold_sensitivity_plots(sensitivity, args.output_dir)
    save_bootstrap_effect_plots(bootstrap, args.output_dir)
    if not shap_context.empty:
        save_shap_regime_plots(shap_context, args.output_dir)

    print(f"File scritto: {classified_path}")
    print(f"File scritto: {summary_path}")
    print(f"File scritto: {regression_path}")
    print(f"File scritto: {effect_path}")
    print(f"File scritto: {quantile_path}")
    print(f"File scritto: {distributed_lag_path}")
    print(f"File scritto: {distributed_lag_effect_path}")
    print(f"File scritto: {matches_path}")
    print(f"File scritto: {matched_summary_path}")
    print(f"File scritto: {bootstrap_path}")
    print(f"File scritto: {sensitivity_path}")
    print(f"File scritto: {station_wind_path}")
    print(f"File scritto: {spatial_gradient_path}")
    print(f"File scritto: {spatial_gradient_summary_path}")
    print(f"File scritto: {did_regression_path}")
    print(f"File scritto: {did_summary_path}")
    print(f"File scritto: {panel_path}")
    if plots_dir.exists():
        print(f"Cartella grafici: {plots_dir}")
    if not shap_context.empty:
        print(f"File scritto: {shap_path}")
    print("\nConteggio regimi vento")
    print(df["wind_regime"].value_counts().to_string())
    print("\nCoefficienti chiave")
    print(effects.to_string(index=False))
    print("\nMatching downwind-upwind")
    print(match_summary.to_string(index=False))
    print("\nBootstrap effetti")
    print(bootstrap.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
