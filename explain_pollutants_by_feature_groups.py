#!/usr/bin/env python3
"""Explain pollutant predictions by traffic, airport and weather feature groups."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor, XGBRegressor
from analysis_runtime import resolve_workers

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    tqdm = None


DEFAULT_INPUT = Path("Datasets_Raw/hourly_merged_2023_2025.csv")
DEFAULT_OUTPUT_DIR = Path("Analysis")
DATETIME_COLUMN = "datetime"
PLOT_FORMAT = "svg"
AIRPORT_REFERENCE = (44.5354, 11.2887)
STATIONS = {
    "porta_san_felice": {"lat": 44.5013, "lon": 11.3280},
    "giardini_margherita": {"lat": 44.4849, "lon": 11.3546},
    "via_chiarini": {"lat": 44.4946, "lon": 11.3768},
}
POLLUTANTS = [
    "NO2_porta_san_felice",
    "CO_porta_san_felice",
    "C6H6_porta_san_felice",
    "NO2_giardini_margherita",
    "NO2_via_chiarini",
    "O3_giardini_margherita",
    "O3_via_chiarini",
]
DEFAULT_HORIZONS = [1, 3, 6, 12, 24]
ADVANCED_LAGS = [1, 2, 3, 6, 12, 24]
ADVANCED_ROLLING_WINDOWS = [3, 6, 12, 24]
STATION_IDS = ["porta_san_felice", "giardini_margherita", "via_chiarini"]
EXTENDED_ABLATION_BOOTSTRAP_REPEATS = 500
FEATURE_GROUP_PRIORITY = [
    "target_autoregressive",
    "pollutant_context",
    "other_pollutants",
    "meteo",
    "wind_transport",
    "airport_wind_interaction",
    "airport_service_type",
    "airport",
    "urban_traffic",
    "station_wind_bools",
    "time",
    "lag_features",
    "rolling_features",
    "diff_features",
]


@dataclass(frozen=True)
class TemporalFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_hours: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valuta quanto traffico urbano, aeroporto e meteo spiegano ogni inquinante "
            "con validazione temporale, feature lag/rolling avanzate e SHAP a gruppi."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Dataset orario unito (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Cartella output report (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=POLLUTANTS,
        help=f"Inquinanti target (default: {' '.join(POLLUTANTS)}).",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=DEFAULT_HORIZONS,
        help=f"Orizzonti di previsione in ore (default: {' '.join(map(str, DEFAULT_HORIZONS))}).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Numero alberi XGBoost (default: 300).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Profondità massima XGBoost (default: 4).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate XGBoost (default: 0.05).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disabilita le barre di progresso.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "ridge",
            "elasticnet",
            "decision_tree",
            "random_forest",
            "extra_trees",
            "adaboost",
            "xgbrf",
            "xgboost",
        ],
        default=["ridge", "decision_tree", "random_forest", "extra_trees", "adaboost", "xgbrf", "xgboost"],
        help="Modelli da confrontare.",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Fold temporali.")
    parser.add_argument("--test-days", type=int, default=45, help="Giorni di test per fold.")
    parser.add_argument("--min-train-days", type=int, default=180, help="Minimo train iniziale.")
    parser.add_argument("--shap-sample", type=int, default=1000, help="Campione test per SHAP.")
    parser.add_argument("--no-shap", action="store_true", help="Disabilita SHAP.")
    parser.add_argument(
        "--no-multioutput-xgb",
        action="store_true",
        help="Disabilita l'esperimento multitarget basato su XGBoost.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Parallelismo locale. 0=auto: usa SLURM_CPUS_PER_TASK se presente, "
            "altrimenti tutte le CPU locali."
        ),
    )
    parser.add_argument(
        "--wind-angle-tolerance-deg",
        type=float,
        default=30.0,
        help="Tolleranza angolare per i booleani downwind/upwind rispetto alla direttrice aeroporto->stazione.",
    )
    parser.add_argument(
        "--wind-bool-min-intensity",
        type=float,
        default=0.5,
        help="Intensita' minima del vento per attivare i booleani downwind/upwind.",
    )
    return parser.parse_args()


class Progress:
    def __init__(self, total: int, enabled: bool, description: str) -> None:
        self.total = total
        self.enabled = enabled
        self.count = 0
        self.description = description
        self.bar = tqdm(total=total, desc=description, unit="step") if enabled and tqdm else None

    def update(self, label: str) -> None:
        self.count += 1
        if self.bar is not None:
            self.bar.set_postfix_str(label[:80])
            self.bar.update(1)
            return

        if self.enabled:
            print(f"[{self.count}/{self.total}] {self.description}: {label}", flush=True)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    df = pd.read_csv(path)
    if DATETIME_COLUMN not in df.columns:
        raise ValueError(f"Colonna {DATETIME_COLUMN!r} mancante")

    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], errors="coerce")
    if df[DATETIME_COLUMN].isna().any():
        raise ValueError(f"Date non valide nella colonna {DATETIME_COLUMN}")

    return df.sort_values(DATETIME_COLUMN).reset_index(drop=True)


def metrics(y_true: pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    y_pred_series = pd.Series(y_pred, index=y_true.index)
    non_zero_mask = y_true != 0
    mape = (
        ((y_true[non_zero_mask] - y_pred_series[non_zero_mask]).abs() / y_true[non_zero_mask])
        .mean()
        * 100
    )
    return {
        "MAE": mean_absolute_error(y_true, y_pred_series),
        "RMSE": mean_squared_error(y_true, y_pred_series) ** 0.5,
        "MAPE_%": mape,
        "R2": r2_score(y_true, y_pred_series),
    }


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def split_pollutant_station(column: str) -> tuple[str, str] | None:
    for station in sorted(STATIONS, key=len, reverse=True):
        suffix = f"_{station}"
        if column.endswith(suffix):
            pollutant = column[: -len(suffix)]
            if pollutant:
                return pollutant, station
    return None


def build_pollutant_station_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for column in df.columns:
        parsed = split_pollutant_station(column)
        if parsed is None or not pd.api.types.is_numeric_dtype(df[column]):
            continue

        pollutant, station = parsed
        values = df[column].dropna()
        rows.append(
            {
                "target": column,
                "pollutant": pollutant,
                "station": station,
                "n_observations": int(values.shape[0]),
                "mean": float(values.mean()) if not values.empty else np.nan,
                "min": float(values.min()) if not values.empty else np.nan,
                "max": float(values.max()) if not values.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["station", "pollutant"]).reset_index(drop=True)


def advanced_traffic_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if c.startswith("blq_") or c.startswith("spire_")]


def advanced_meteo_columns(columns: list[str]) -> list[str]:
    return [
        c
        for c in columns
        if c.endswith("_aero")
        or c.endswith("_centro")
        or c.startswith("airport_to_psf_")
    ]


def initial_bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    delta_lambda = np.deg2rad(lon2 - lon1)
    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    return float((np.rad2deg(np.arctan2(y, x)) + 360) % 360)


def signed_angle_diff_degrees(direction: pd.Series, target_direction: float) -> pd.Series:
    return (direction - target_direction + 180) % 360 - 180


def add_station_wind_boolean_features(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    for station_id, station in STATIONS.items():
        bearing = initial_bearing_degrees(
            AIRPORT_REFERENCE[0],
            AIRPORT_REFERENCE[1],
            station["lat"],
            station["lon"],
        )
        opposite_bearing = (bearing + 180.0) % 360.0
        for source in ["aero", "centro"]:
            direction_col = f"W_VEC_DIR_{source}"
            intensity_col = f"W_VEC_INT_{source}"
            if direction_col not in out.columns or intensity_col not in out.columns:
                continue
            wind_to_direction = (out[direction_col] + 180.0) % 360.0
            down_diff = signed_angle_diff_degrees(wind_to_direction, bearing).abs()
            up_diff = signed_angle_diff_degrees(wind_to_direction, opposite_bearing).abs()
            moving = out[intensity_col].abs() >= args.wind_bool_min_intensity
            out[f"airport_to_{station_id}_downwind_bool_{source}"] = (
                moving & (down_diff <= args.wind_angle_tolerance_deg)
            ).astype(int)
            out[f"airport_to_{station_id}_upwind_bool_{source}"] = (
                moving & (up_diff <= args.wind_angle_tolerance_deg)
            ).astype(int)
    return out


def advanced_time_features(df: pd.DataFrame) -> list[pd.Series]:
    dt = df[DATETIME_COLUMN]
    hour = dt.dt.hour
    month = dt.dt.month
    day = dt.dt.dayofweek
    return [
        hour.rename("hour"),
        day.rename("dayofweek"),
        month.rename("month"),
        (day >= 5).astype(int).rename("is_weekend"),
        np.sin(2 * np.pi * hour / 24).rename("hour_sin"),
        np.cos(2 * np.pi * hour / 24).rename("hour_cos"),
        np.sin(2 * np.pi * month / 12).rename("month_sin"),
        np.cos(2 * np.pi * month / 12).rename("month_cos"),
    ]


def advanced_lag_rolling_features(df: pd.DataFrame, columns: list[str]) -> list[pd.Series]:
    features: list[pd.Series] = []
    for column in columns:
        for lag in ADVANCED_LAGS:
            features.append(df[column].shift(lag).rename(f"{column}_lag_{lag}h"))
        features.append((df[column] - df[column].shift(1)).rename(f"{column}_diff_1h"))
        shifted = df[column].shift(1)
        for window in ADVANCED_ROLLING_WINDOWS:
            min_periods = max(1, window // 2)
            features.append(
                shifted.rolling(window=window, min_periods=min_periods)
                .mean()
                .rename(f"{column}_rolling_{window}h_mean")
            )
            features.append(
                shifted.rolling(window=window, min_periods=min_periods)
                .std()
                .rename(f"{column}_rolling_{window}h_std")
            )
    return features


def advanced_interaction_features(df: pd.DataFrame) -> list[pd.Series]:
    features: list[pd.Series] = []
    wind_cols = [
        "airport_to_psf_wind_alignment_aero",
        "airport_to_psf_wind_component_aero",
        "airport_to_psf_wind_alignment_centro",
        "airport_to_psf_wind_component_centro",
    ]
    source_cols = [c for c in df.columns if c.startswith("blq_") or c.startswith("spire_airport_")]
    for source in source_cols:
        for wind in wind_cols:
            if wind in df.columns:
                features.append((df[source] * df[wind]).rename(f"{source}_x_{wind}"))
    return features


def build_advanced_base_frame(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    base = add_station_wind_boolean_features(df, args)
    parts: list[pd.Series] = []
    parts.extend(advanced_time_features(base))
    present_pollutants = [p for p in POLLUTANTS if p in base.columns]
    parts.extend(advanced_lag_rolling_features(base, present_pollutants))
    parts.extend(advanced_lag_rolling_features(base, advanced_traffic_columns(list(base.columns))))
    parts.extend(advanced_lag_rolling_features(base, advanced_meteo_columns(list(base.columns))))
    parts.extend(advanced_interaction_features(base))
    return pd.concat([base, *parts], axis=1).copy()


def advanced_feature_frame(base_df: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    out = base_df.copy()
    out[f"target_{target}"] = out[target].shift(-horizon)
    return out


def timing_row(
    scope: str,
    step: str,
    seconds: float,
    target: str | None = None,
    horizon: int | None = None,
) -> dict[str, object]:
    return {
        "scope": scope,
        "step": step,
        "target": target,
        "horizon_h": horizon,
        "seconds": float(seconds),
    }


def classify_feature(feature: str) -> dict[str, object]:
    return {
        "is_lag": "_lag_" in feature,
        "is_rolling": "_rolling_" in feature,
        "is_diff": "_diff_" in feature,
        "is_interaction": "_x_" in feature,
        "is_airport_service_type": feature.startswith("blq_service_"),
        "is_station_wind_bool": "_downwind_bool_" in feature or "_upwind_bool_" in feature,
        "is_calendar": feature
        in {"hour", "dayofweek", "month", "is_weekend", "hour_sin", "hour_cos", "month_sin", "month_cos"},
    }


def primary_group_for_feature(feature: str, groups: dict[str, list[str]]) -> str:
    memberships = {group for group, values in groups.items() if feature in values}
    for group in FEATURE_GROUP_PRIORITY:
        if group in memberships:
            return group
    return sorted(memberships)[0] if memberships else "unassigned"


def advanced_feature_groups(columns: list[str], target: str) -> dict[str, list[str]]:
    urban_spire_columns = [c for c in columns if c.startswith("spire_") and not c.startswith("spire_airport_")]
    target_auto = [
        c
        for c in columns
        if c == target
        or c.startswith(f"{target}_lag_")
        or c.startswith(f"{target}_rolling_")
        or c.startswith(f"{target}_diff_")
    ]
    service_type = [c for c in columns if c.startswith("blq_service_")]
    station_wind_bools = [c for c in columns if "_downwind_bool_" in c or "_upwind_bool_" in c]
    lag_features = [c for c in columns if "_lag_" in c]
    rolling_features = [c for c in columns if "_rolling_" in c]
    diff_features = [c for c in columns if "_diff_" in c]
    other_pollutants_by_station = {
        station_id: [
            c
            for p in POLLUTANTS
            if p != target and station_id in p
            for c in columns
            if c == p or c.startswith(f"{p}_lag_") or c.startswith(f"{p}_rolling_") or c.startswith(f"{p}_diff_")
        ]
        for station_id in STATION_IDS
    }
    return {
        "time": [
            c
            for c in columns
            if c in {"hour", "dayofweek", "month", "is_weekend", "hour_sin", "hour_cos", "month_sin", "month_cos"}
        ],
        "meteo": [
            c
            for c in columns
            if (
                c.endswith("_aero")
                or c.endswith("_centro")
                or "_aero_" in c
                or "_centro_" in c
            )
            and not c.startswith("spire_")
        ],
        "wind_transport": [
            c
            for c in columns
            if (c.startswith("airport_to_psf_") or "_airport_to_psf_" in c)
            and c not in station_wind_bools
        ],
        "airport": [
            c
            for c in columns
            if (c.startswith("blq_") and not c.startswith("blq_service_")) or c.startswith("spire_airport_")
        ],
        "airport_service_type": service_type,
        "urban_traffic": urban_spire_columns,
        "airport_wind_interaction": [c for c in columns if "_x_airport_to_psf_" in c],
        "station_wind_bools": station_wind_bools,
        "other_pollutants": [
            c
            for p in POLLUTANTS
            if p != target
            for c in columns
            if c == p or c.startswith(f"{p}_lag_") or c.startswith(f"{p}_rolling_") or c.startswith(f"{p}_diff_")
        ],
        "target_autoregressive": target_auto,
        "lag_features": lag_features,
        "rolling_features": rolling_features,
        "diff_features": diff_features,
        **{f"other_pollutants_{station_id}": values for station_id, values in other_pollutants_by_station.items()},
    }


def advanced_feature_sets(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    full_no_auto = unique_preserve_order(
        groups["time"]
        + groups["meteo"]
        + groups["wind_transport"]
        + groups["airport"]
        + groups["airport_service_type"]
        + groups["urban_traffic"]
        + groups["airport_wind_interaction"]
        + groups["station_wind_bools"]
        + groups["other_pollutants"]
    )
    no_service_type = [c for c in full_no_auto if c not in groups["airport_service_type"]]
    no_station_wind_bools = [c for c in full_no_auto if c not in groups["station_wind_bools"]]
    no_service_type_or_station_wind_bools = [
        c for c in full_no_auto if c not in groups["airport_service_type"] and c not in groups["station_wind_bools"]
    ]
    return {
        "no_target_autoregressive": full_no_auto,
        "no_target_without_service_type": no_service_type,
        "no_target_without_station_wind_bools": no_station_wind_bools,
        "no_target_without_service_type_or_station_wind_bools": no_service_type_or_station_wind_bools,
        "with_target_autoregressive": unique_preserve_order(full_no_auto + groups["target_autoregressive"]),
        "with_target_without_service_type": unique_preserve_order(no_service_type + groups["target_autoregressive"]),
        "with_target_without_station_wind_bools": unique_preserve_order(
            no_station_wind_bools + groups["target_autoregressive"]
        ),
        "with_target_without_service_type_or_station_wind_bools": unique_preserve_order(
            no_service_type_or_station_wind_bools + groups["target_autoregressive"]
        ),
    }


def target_output_column(target: str) -> str:
    return f"target__{target}"


def multioutput_feature_frame(base_df: pd.DataFrame, targets: list[str], horizon: int) -> pd.DataFrame:
    out = base_df.copy()
    for target in targets:
        out[target_output_column(target)] = out[target].shift(-horizon)
    return out


def multioutput_feature_groups(columns: list[str], targets: list[str]) -> dict[str, list[str]]:
    urban_spire_columns = [c for c in columns if c.startswith("spire_") and not c.startswith("spire_airport_")]
    pollutant_context = [
        c
        for target in targets
        for c in columns
        if c == target
        or c.startswith(f"{target}_lag_")
        or c.startswith(f"{target}_rolling_")
        or c.startswith(f"{target}_diff_")
    ]
    service_type = [c for c in columns if c.startswith("blq_service_")]
    station_wind_bools = [c for c in columns if "_downwind_bool_" in c or "_upwind_bool_" in c]
    lag_features = [c for c in columns if "_lag_" in c]
    rolling_features = [c for c in columns if "_rolling_" in c]
    diff_features = [c for c in columns if "_diff_" in c]
    pollutant_context_by_station = {
        station_id: [
            c
            for target in targets
            if station_id in target
            for c in columns
            if c == target or c.startswith(f"{target}_lag_") or c.startswith(f"{target}_rolling_") or c.startswith(f"{target}_diff_")
        ]
        for station_id in STATION_IDS
    }
    return {
        "time": [
            c
            for c in columns
            if c in {"hour", "dayofweek", "month", "is_weekend", "hour_sin", "hour_cos", "month_sin", "month_cos"}
        ],
        "meteo": [
            c
            for c in columns
            if (
                c.endswith("_aero")
                or c.endswith("_centro")
                or "_aero_" in c
                or "_centro_" in c
            )
            and not c.startswith("spire_")
        ],
        "wind_transport": [
            c
            for c in columns
            if (c.startswith("airport_to_psf_") or "_airport_to_psf_" in c)
            and c not in station_wind_bools
        ],
        "airport": [
            c
            for c in columns
            if (c.startswith("blq_") and not c.startswith("blq_service_")) or c.startswith("spire_airport_")
        ],
        "airport_service_type": service_type,
        "urban_traffic": urban_spire_columns,
        "airport_wind_interaction": [c for c in columns if "_x_airport_to_psf_" in c],
        "station_wind_bools": station_wind_bools,
        "pollutant_context": pollutant_context,
        "lag_features": lag_features,
        "rolling_features": rolling_features,
        "diff_features": diff_features,
        **{f"pollutant_context_{station_id}": values for station_id, values in pollutant_context_by_station.items()},
    }


def multioutput_feature_sets(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    no_context = unique_preserve_order(
        groups["time"]
        + groups["meteo"]
        + groups["wind_transport"]
        + groups["airport"]
        + groups["airport_service_type"]
        + groups["urban_traffic"]
        + groups["airport_wind_interaction"]
        + groups["station_wind_bools"]
    )
    no_context_without_service_type = [c for c in no_context if c not in groups["airport_service_type"]]
    no_context_without_station_wind_bools = [c for c in no_context if c not in groups["station_wind_bools"]]
    no_context_without_service_type_or_station_wind_bools = [
        c for c in no_context if c not in groups["airport_service_type"] and c not in groups["station_wind_bools"]
    ]
    return {
        "no_pollutant_context": no_context,
        "no_pollutant_context_without_service_type": no_context_without_service_type,
        "no_pollutant_context_without_station_wind_bools": no_context_without_station_wind_bools,
        "no_pollutant_context_without_service_type_or_station_wind_bools": no_context_without_service_type_or_station_wind_bools,
        "with_pollutant_context": unique_preserve_order(no_context + groups["pollutant_context"]),
        "with_pollutant_context_without_service_type": unique_preserve_order(
            no_context_without_service_type + groups["pollutant_context"]
        ),
        "with_pollutant_context_without_station_wind_bools": unique_preserve_order(
            no_context_without_station_wind_bools + groups["pollutant_context"]
        ),
        "with_pollutant_context_without_service_type_or_station_wind_bools": unique_preserve_order(
            no_context_without_service_type_or_station_wind_bools + groups["pollutant_context"]
        ),
    }


def register_feature_set(
    feature_sets: dict[str, list[str]],
    metadata_rows: list[dict],
    name: str,
    features: list[str],
    meta: dict,
    seen_signatures: set[tuple[str, ...]],
) -> None:
    cleaned = unique_preserve_order(features)
    if not cleaned:
        return
    signature = tuple(cleaned)
    if signature in seen_signatures:
        return
    seen_signatures.add(signature)
    feature_sets[name] = cleaned
    metadata_rows.append({"feature_set": name, "n_features": len(cleaned), **meta})


def drop_groups(features: list[str], groups: dict[str, list[str]], group_names: list[str]) -> list[str]:
    to_remove = set()
    for group_name in group_names:
        to_remove.update(groups.get(group_name, []))
    return [feature for feature in features if feature not in to_remove]


def cumulative_features(groups: dict[str, list[str]], block_order: list[str]) -> list[list[str]]:
    built: list[list[str]] = []
    current: list[str] = []
    for block in block_order:
        current = unique_preserve_order(current + groups.get(block, []))
        built.append(current.copy())
    return built


def build_single_extended_ablation_feature_sets(groups: dict[str, list[str]]) -> tuple[dict[str, list[str]], pd.DataFrame]:
    feature_sets: dict[str, list[str]] = {}
    metadata_rows: list[dict] = []
    seen_signatures: set[tuple[str, ...]] = set()

    base_sets = advanced_feature_sets(groups)
    baselines = {
        "no_target_autoregressive": base_sets["no_target_autoregressive"],
        "with_target_autoregressive": base_sets["with_target_autoregressive"],
    }

    removable_groups = [
        "time",
        "meteo",
        "wind_transport",
        "airport",
        "airport_service_type",
        "urban_traffic",
        "airport_wind_interaction",
        "station_wind_bools",
        "other_pollutants",
        "lag_features",
        "rolling_features",
        "diff_features",
    ]
    if groups.get("target_autoregressive"):
        removable_groups.append("target_autoregressive")
    removable_groups.extend(
        [group_name for group_name in groups if group_name.startswith("other_pollutants_") and groups[group_name]]
    )

    pairwise_groups = [
        ("airport", "urban_traffic"),
        ("airport_service_type", "airport_wind_interaction"),
        ("meteo", "wind_transport"),
        ("airport", "airport_service_type"),
        ("airport", "station_wind_bools"),
        ("airport_service_type", "station_wind_bools"),
        ("airport", "wind_transport"),
        ("lag_features", "rolling_features"),
        ("other_pollutants", "target_autoregressive"),
    ]

    ladder_blocks_no_target = [
        "time",
        "meteo",
        "urban_traffic",
        "airport",
        "airport_service_type",
        "wind_transport",
        "airport_wind_interaction",
        "station_wind_bools",
        "other_pollutants",
    ]
    ladder_blocks_with_target = ladder_blocks_no_target + ["target_autoregressive"]

    only_block_specs = {
        "only_time": ["time"],
        "only_meteo": ["meteo"],
        "only_airport": ["airport", "airport_service_type"],
        "only_airport_service_type": ["airport_service_type"],
        "only_urban_traffic": ["urban_traffic"],
        "only_wind_transport": ["wind_transport", "station_wind_bools", "airport_wind_interaction"],
        "only_station_wind_bools": ["station_wind_bools"],
        "only_other_pollutants": ["other_pollutants"],
        "only_target_autoregressive": ["target_autoregressive"],
        "only_meteo_plus_urban_traffic": ["meteo", "urban_traffic"],
        "only_airport_plus_meteo": ["airport", "airport_service_type", "meteo"],
        "only_airport_plus_urban_traffic": ["airport", "airport_service_type", "urban_traffic"],
    }

    for baseline_name, baseline_features in baselines.items():
        register_feature_set(
            feature_sets,
            metadata_rows,
            baseline_name,
            baseline_features,
            {
                "analysis_scope": "single_target",
                "analysis_type": "baseline",
                "baseline_feature_set": baseline_name,
            },
            seen_signatures,
        )
        for group_name in removable_groups:
            if not groups.get(group_name):
                continue
            candidate = drop_groups(baseline_features, groups, [group_name])
            register_feature_set(
                feature_sets,
                metadata_rows,
                f"{baseline_name}__drop__{group_name}",
                candidate,
                {
                    "analysis_scope": "single_target",
                    "analysis_type": "drop_one_group",
                    "baseline_feature_set": baseline_name,
                    "removed_groups": group_name,
                },
                seen_signatures,
            )
        for first_group, second_group in pairwise_groups:
            if not groups.get(first_group) or not groups.get(second_group):
                continue
            if first_group == "target_autoregressive" and baseline_name == "no_target_autoregressive":
                continue
            if second_group == "target_autoregressive" and baseline_name == "no_target_autoregressive":
                continue
            removed = [first_group, second_group]
            candidate = drop_groups(baseline_features, groups, removed)
            register_feature_set(
                feature_sets,
                metadata_rows,
                f"{baseline_name}__drop__{first_group}__{second_group}",
                candidate,
                {
                    "analysis_scope": "single_target",
                    "analysis_type": "drop_pair",
                    "baseline_feature_set": baseline_name,
                    "removed_groups": "|".join(removed),
                },
                seen_signatures,
            )

    for ladder_name, block_order in [
        ("no_target_ladder", ladder_blocks_no_target),
        ("with_target_ladder", ladder_blocks_with_target),
    ]:
        previous_name = ""
        for stage_idx, features in enumerate(cumulative_features(groups, block_order), start=1):
            block = block_order[stage_idx - 1]
            name = f"{ladder_name}__stage_{stage_idx:02d}__{block}"
            register_feature_set(
                feature_sets,
                metadata_rows,
                name,
                features,
                {
                    "analysis_scope": "single_target",
                    "analysis_type": "cumulative",
                    "ladder_name": ladder_name,
                    "stage_order": stage_idx,
                    "stage_block": block,
                    "previous_feature_set": previous_name,
                },
                seen_signatures,
            )
            previous_name = name

    for spec_name, block_names in only_block_specs.items():
        selected = unique_preserve_order([feature for block in block_names for feature in groups.get(block, [])])
        register_feature_set(
            feature_sets,
            metadata_rows,
            spec_name,
            selected,
            {
                "analysis_scope": "single_target",
                "analysis_type": "only_block",
                "block_combo": "|".join(block_names),
            },
            seen_signatures,
        )

    return feature_sets, pd.DataFrame(metadata_rows)


def build_multioutput_extended_ablation_feature_sets(groups: dict[str, list[str]]) -> tuple[dict[str, list[str]], pd.DataFrame]:
    feature_sets: dict[str, list[str]] = {}
    metadata_rows: list[dict] = []
    seen_signatures: set[tuple[str, ...]] = set()

    base_sets = multioutput_feature_sets(groups)
    baselines = {
        "no_pollutant_context": base_sets["no_pollutant_context"],
        "with_pollutant_context": base_sets["with_pollutant_context"],
    }

    removable_groups = [
        "time",
        "meteo",
        "wind_transport",
        "airport",
        "airport_service_type",
        "urban_traffic",
        "airport_wind_interaction",
        "station_wind_bools",
        "lag_features",
        "rolling_features",
        "diff_features",
    ]
    if groups.get("pollutant_context"):
        removable_groups.append("pollutant_context")
    removable_groups.extend(
        [group_name for group_name in groups if group_name.startswith("pollutant_context_") and groups[group_name]]
    )

    pairwise_groups = [
        ("airport", "urban_traffic"),
        ("airport_service_type", "airport_wind_interaction"),
        ("meteo", "wind_transport"),
        ("airport", "airport_service_type"),
        ("airport", "station_wind_bools"),
        ("airport_service_type", "station_wind_bools"),
        ("airport", "wind_transport"),
        ("lag_features", "rolling_features"),
        ("pollutant_context", "station_wind_bools"),
    ]

    ladder_blocks_no_context = [
        "time",
        "meteo",
        "urban_traffic",
        "airport",
        "airport_service_type",
        "wind_transport",
        "airport_wind_interaction",
        "station_wind_bools",
    ]
    ladder_blocks_with_context = ladder_blocks_no_context + ["pollutant_context"]

    only_block_specs = {
        "only_time": ["time"],
        "only_meteo": ["meteo"],
        "only_airport": ["airport", "airport_service_type"],
        "only_airport_service_type": ["airport_service_type"],
        "only_urban_traffic": ["urban_traffic"],
        "only_wind_transport": ["wind_transport", "station_wind_bools", "airport_wind_interaction"],
        "only_station_wind_bools": ["station_wind_bools"],
        "only_pollutant_context": ["pollutant_context"],
        "only_meteo_plus_urban_traffic": ["meteo", "urban_traffic"],
        "only_airport_plus_meteo": ["airport", "airport_service_type", "meteo"],
        "only_airport_plus_urban_traffic": ["airport", "airport_service_type", "urban_traffic"],
    }

    for baseline_name, baseline_features in baselines.items():
        register_feature_set(
            feature_sets,
            metadata_rows,
            baseline_name,
            baseline_features,
            {
                "analysis_scope": "multioutput",
                "analysis_type": "baseline",
                "baseline_feature_set": baseline_name,
            },
            seen_signatures,
        )
        for group_name in removable_groups:
            if not groups.get(group_name):
                continue
            if group_name == "pollutant_context" and baseline_name == "no_pollutant_context":
                continue
            if group_name.startswith("pollutant_context_") and baseline_name == "no_pollutant_context":
                continue
            candidate = drop_groups(baseline_features, groups, [group_name])
            analysis_type = "leave_one_station_out_context" if group_name.startswith("pollutant_context_") else "drop_one_group"
            register_feature_set(
                feature_sets,
                metadata_rows,
                f"{baseline_name}__drop__{group_name}",
                candidate,
                {
                    "analysis_scope": "multioutput",
                    "analysis_type": analysis_type,
                    "baseline_feature_set": baseline_name,
                    "removed_groups": group_name,
                },
                seen_signatures,
            )
        for first_group, second_group in pairwise_groups:
            if not groups.get(first_group) or not groups.get(second_group):
                continue
            if first_group == "pollutant_context" and baseline_name == "no_pollutant_context":
                continue
            if second_group == "pollutant_context" and baseline_name == "no_pollutant_context":
                continue
            removed = [first_group, second_group]
            candidate = drop_groups(baseline_features, groups, removed)
            register_feature_set(
                feature_sets,
                metadata_rows,
                f"{baseline_name}__drop__{first_group}__{second_group}",
                candidate,
                {
                    "analysis_scope": "multioutput",
                    "analysis_type": "drop_pair",
                    "baseline_feature_set": baseline_name,
                    "removed_groups": "|".join(removed),
                },
                seen_signatures,
            )

    for ladder_name, block_order in [
        ("no_context_ladder", ladder_blocks_no_context),
        ("with_context_ladder", ladder_blocks_with_context),
    ]:
        previous_name = ""
        for stage_idx, features in enumerate(cumulative_features(groups, block_order), start=1):
            block = block_order[stage_idx - 1]
            name = f"{ladder_name}__stage_{stage_idx:02d}__{block}"
            register_feature_set(
                feature_sets,
                metadata_rows,
                name,
                features,
                {
                    "analysis_scope": "multioutput",
                    "analysis_type": "cumulative",
                    "ladder_name": ladder_name,
                    "stage_order": stage_idx,
                    "stage_block": block,
                    "previous_feature_set": previous_name,
                },
                seen_signatures,
            )
            previous_name = name

    for spec_name, block_names in only_block_specs.items():
        selected = unique_preserve_order([feature for block in block_names for feature in groups.get(block, [])])
        register_feature_set(
            feature_sets,
            metadata_rows,
            spec_name,
            selected,
            {
                "analysis_scope": "multioutput",
                "analysis_type": "only_block",
                "block_combo": "|".join(block_names),
            },
            seen_signatures,
        )

    return feature_sets, pd.DataFrame(metadata_rows)


def temporal_folds(df: pd.DataFrame, args: argparse.Namespace, horizon: int) -> list[TemporalFold]:
    start = df[DATETIME_COLUMN].min()
    end = df[DATETIME_COLUMN].max()
    min_train_end = start + pd.Timedelta(days=args.min_train_days)
    test_delta = pd.Timedelta(days=args.test_days)
    latest_test_start = end - test_delta + pd.Timedelta(hours=1)
    if latest_test_start <= min_train_end:
        raise ValueError("Periodo troppo corto per la configurazione di validazione temporale.")
    starts = pd.date_range(min_train_end, latest_test_start, periods=args.cv_folds)
    return [
        TemporalFold(
            fold=i,
            train_start=start,
            train_end=pd.Timestamp(test_start).floor("h") - pd.Timedelta(hours=horizon + 1),
            test_start=pd.Timestamp(test_start).floor("h"),
            test_end=min(pd.Timestamp(test_start).floor("h") + test_delta - pd.Timedelta(hours=1), end),
            embargo_hours=horizon,
        )
        for i, test_start in enumerate(starts, start=1)
    ]


def build_advanced_model(name: str, features: list[str], args: argparse.Namespace) -> Pipeline:
    preprocess = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if name == "ridge":
        steps = [("preprocess", preprocess), ("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]
    elif name == "elasticnet":
        steps = [
            ("preprocess", preprocess),
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=0.1, l1_ratio=0.2, max_iter=5000, tol=0.01, selection="random", random_state=42)),
        ]
    elif name == "decision_tree":
        steps = [
            ("preprocess", preprocess),
            ("model", DecisionTreeRegressor(max_depth=args.max_depth, min_samples_leaf=24, random_state=42)),
        ]
    elif name == "random_forest":
        steps = [
            ("preprocess", preprocess),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    min_samples_leaf=12,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    elif name == "extra_trees":
        steps = [
            ("preprocess", preprocess),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    min_samples_leaf=12,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    elif name == "adaboost":
        steps = [
            ("preprocess", preprocess),
            (
                "model",
                AdaBoostRegressor(
                    estimator=DecisionTreeRegressor(
                        max_depth=max(2, min(args.max_depth, 6)),
                        min_samples_leaf=12,
                        random_state=42,
                    ),
                    n_estimators=args.n_estimators,
                    learning_rate=max(args.learning_rate, 0.05),
                    random_state=42,
                ),
            ),
        ]
    elif name == "xgbrf":
        steps = [
            ("preprocess", preprocess),
            (
                "model",
                XGBRFRegressor(
                    random_state=42,
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    learning_rate=1.0,
                    subsample=0.8,
                    colsample_bynode=0.8,
                    objective="reg:squarederror",
                    n_jobs=1,
                ),
            ),
        ]
    elif name == "xgboost":
        steps = [
            ("preprocess", preprocess),
            (
                "model",
                XGBRegressor(
                    random_state=42,
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    learning_rate=args.learning_rate,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    n_jobs=1,
                ),
            ),
        ]
    else:
        raise ValueError(f"Modello non supportato: {name}")
    return Pipeline(steps)


def build_multioutput_xgboost(features: list[str], args: argparse.Namespace) -> Pipeline:
    preprocess = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    estimator = XGBRegressor(
        random_state=42,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=1,
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", MultiOutputRegressor(estimator, n_jobs=1)),
        ]
    )


def import_shap():
    try:
        import shap as shap_module
    except ImportError:  # pragma: no cover
        return None
    return shap_module


def pipeline_transform_features(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if len(model.steps) == 1:
        return X.to_numpy()
    return model[:-1].transform(X)


def pipeline_predict(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    transformed = pipeline_transform_features(model, X)
    return model.named_steps["model"].predict(transformed)


def evaluate_persistence_baselines(
    base_df: pd.DataFrame, target: str, horizon: int, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_col = f"target_{target}"
    data = advanced_feature_frame(base_df[[DATETIME_COLUMN, target]].copy(), target, horizon)
    data["naive_persistence"] = data[target]
    data["seasonal_24h_persistence"] = data[target].shift(24 - horizon)
    data["rolling_24h_mean"] = data[target].shift(1).rolling(window=24, min_periods=12).mean()
    data = data.dropna(subset=[target_col]).reset_index(drop=True)
    folds = temporal_folds(data, args, horizon)
    baseline_columns = {
        "naive_persistence": "naive_persistence",
        "seasonal_24h_persistence": "seasonal_24h_persistence",
        "rolling_24h_mean": "rolling_24h_mean",
    }
    rows: list[dict] = []
    prediction_rows: list[dict] = []
    for fold in folds:
        test = data.loc[(data[DATETIME_COLUMN] >= fold.test_start) & (data[DATETIME_COLUMN] <= fold.test_end)]
        for baseline_name, pred_col in baseline_columns.items():
            valid = test.dropna(subset=[pred_col, target_col])
            if valid.empty:
                continue
            pred = valid[pred_col]
            row = {
                "target": target,
                "horizon_h": horizon,
                "fold": fold.fold,
                "model": baseline_name,
                "feature_set": "persistence_baseline",
                "n_features": 1 if baseline_name != "rolling_24h_mean" else 24,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "embargo_hours": fold.embargo_hours,
                "train_rows": int(((data[DATETIME_COLUMN] >= fold.train_start) & (data[DATETIME_COLUMN] <= fold.train_end)).sum()),
                "test_rows": len(valid),
            }
            row.update(metrics(valid[target_col], pred))
            rows.append(row)
            prediction_rows.extend(
                {
                    "target": target,
                    "horizon_h": horizon,
                    "fold": fold.fold,
                    "model": baseline_name,
                    "feature_set": "persistence_baseline",
                    "datetime": dt,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
                for dt, y_true, y_pred in zip(valid[DATETIME_COLUMN], valid[target_col], pred)
            )
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def build_feature_inventory(
    base_df: pd.DataFrame, targets: list[str], horizons: list[int], args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory_rows: list[dict] = []
    group_rows: list[dict] = []
    fold_rows: list[dict] = []
    present_targets = [target for target in targets if target in base_df.columns]
    for target in present_targets:
        for horizon in horizons:
            target_col = f"target_{target}"
            data = advanced_feature_frame(base_df, target, horizon).dropna(subset=[target, target_col]).reset_index(drop=True)
            feature_candidates = [c for c in data.columns if c not in {DATETIME_COLUMN, target_col}]
            groups = advanced_feature_groups(feature_candidates, target)
            feature_sets = advanced_feature_sets(groups)
            folds = temporal_folds(data, args, horizon)
            for fold in folds:
                fold_rows.append(
                    {
                        "scope": "single_target",
                        "target": target,
                        "horizon_h": horizon,
                        "fold": fold.fold,
                        "train_start": fold.train_start,
                        "train_end": fold.train_end,
                        "test_start": fold.test_start,
                        "test_end": fold.test_end,
                "embargo_hours": fold.embargo_hours,
                        "train_rows": int(((data[DATETIME_COLUMN] >= fold.train_start) & (data[DATETIME_COLUMN] <= fold.train_end)).sum()),
                        "test_rows": int(((data[DATETIME_COLUMN] >= fold.test_start) & (data[DATETIME_COLUMN] <= fold.test_end)).sum()),
                    }
                )
            for group, values in groups.items():
                unique_values = unique_preserve_order(values)
                group_rows.append(
                    {
                        "scope": "single_target",
                        "target": target,
                        "horizon_h": horizon,
                        "group": group,
                        "n_features": len(unique_values),
                    }
                )
            for feature in feature_candidates:
                memberships = sorted(group for group, values in groups.items() if feature in values)
                feature_set_memberships = sorted(name for name, values in feature_sets.items() if feature in values)
                flags = classify_feature(feature)
                row = {
                    "scope": "single_target",
                    "target": target,
                    "horizon_h": horizon,
                    "feature": feature,
                    "primary_group": primary_group_for_feature(feature, groups),
                    "all_groups": ";".join(memberships),
                    "used_in_feature_sets": ";".join(feature_set_memberships),
                    "n_group_memberships": len(memberships),
                    "n_feature_set_memberships": len(feature_set_memberships),
                    "missing_count": int(data[feature].isna().sum()),
                    "missing_pct": float(data[feature].isna().mean() * 100),
                }
                row.update(flags)
                row["is_raw"] = not any(
                    row[key]
                    for key in ["is_lag", "is_rolling", "is_diff", "is_interaction", "is_calendar"]
                )
                inventory_rows.append(row)
    inventory = pd.DataFrame(inventory_rows)
    group_summary = pd.DataFrame(group_rows)
    fold_summary = pd.DataFrame(fold_rows)
    return inventory, group_summary, fold_summary


def evaluate_temporal_cv(
    base_df: pd.DataFrame, target: str, horizon: int, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    total_start = time.perf_counter()
    target_col = f"target_{target}"
    prep_start = time.perf_counter()
    data = advanced_feature_frame(base_df, target, horizon)
    data = data.dropna(subset=[target, target_col]).reset_index(drop=True)
    prep_seconds = time.perf_counter() - prep_start
    setup_start = time.perf_counter()
    feature_candidates = [c for c in data.columns if c not in {DATETIME_COLUMN, target_col}]
    groups = advanced_feature_groups(feature_candidates, target)
    feature_sets = advanced_feature_sets(groups)
    folds = temporal_folds(data, args, horizon)
    setup_seconds = time.perf_counter() - setup_start
    rows: list[dict] = []
    prediction_rows: list[dict] = []
    fitted_for_shap: dict = {}
    progress = Progress(
        total=len(folds) * len(args.models) * len(feature_sets),
        enabled=not args.no_progress,
        description=f"{target} t+{horizon}h",
    )

    cv_start = time.perf_counter()
    for fold in folds:
        train = data.loc[(data[DATETIME_COLUMN] >= fold.train_start) & (data[DATETIME_COLUMN] <= fold.train_end)]
        test = data.loc[(data[DATETIME_COLUMN] >= fold.test_start) & (data[DATETIME_COLUMN] <= fold.test_end)]
        for feature_set_name, features in feature_sets.items():
            for model_name in args.models:
                model = build_advanced_model(model_name, features, args)
                model.fit(train[features], train[target_col])
                pred = pipeline_predict(model, test[features])
                row = {
                    "target": target,
                    "horizon_h": horizon,
                    "fold": fold.fold,
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "n_features": len(features),
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                "embargo_hours": fold.embargo_hours,
                    "train_rows": len(train),
                    "test_rows": len(test),
                }
                row.update(metrics(test[target_col], pred))
                rows.append(row)
                prediction_rows.extend(
                    {
                        "target": target,
                        "horizon_h": horizon,
                        "fold": fold.fold,
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "datetime": dt,
                        "y_true": y_true,
                        "y_pred": y_pred,
                    }
                    for dt, y_true, y_pred in zip(test[DATETIME_COLUMN], test[target_col], pred)
                )
                if model_name == "xgboost" and feature_set_name == "with_target_autoregressive" and fold.fold == folds[-1].fold:
                    fitted_for_shap = {"model": model, "features": features, "test": test, "groups": groups}
                progress.update(f"fold {fold.fold} {model_name} {feature_set_name}")
    progress.close()
    timings = pd.DataFrame(
        [
            timing_row("single_target", "prepare_feature_frame", prep_seconds, target=target, horizon=horizon),
            timing_row("single_target", "prepare_feature_sets", setup_seconds, target=target, horizon=horizon),
            timing_row("single_target", "cross_validation", time.perf_counter() - cv_start, target=target, horizon=horizon),
            timing_row("single_target", "total", time.perf_counter() - total_start, target=target, horizon=horizon),
        ]
    )
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows), fitted_for_shap, timings


def evaluate_multioutput_xgboost(
    base_df: pd.DataFrame, targets: list[str], horizon: int, args: argparse.Namespace
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    total_start = time.perf_counter()
    present_targets = [target for target in targets if target in base_df.columns]
    if len(present_targets) < 2:
        return pd.DataFrame(), {}, pd.DataFrame()

    target_cols = [target_output_column(target) for target in present_targets]
    prep_start = time.perf_counter()
    data = multioutput_feature_frame(base_df, present_targets, horizon)
    data = data.dropna(subset=target_cols).reset_index(drop=True)
    prep_seconds = time.perf_counter() - prep_start
    setup_start = time.perf_counter()
    excluded = {DATETIME_COLUMN, *target_cols}
    feature_candidates = [c for c in data.columns if c not in excluded]
    groups = multioutput_feature_groups(feature_candidates, present_targets)
    feature_sets = multioutput_feature_sets(groups)
    folds = temporal_folds(data, args, horizon)
    setup_seconds = time.perf_counter() - setup_start
    rows: list[dict] = []
    fitted_for_explain: dict = {}
    progress = Progress(
        total=len(folds) * len(feature_sets),
        enabled=not args.no_progress,
        description=f"multioutput_xgboost t+{horizon}h",
    )

    cv_start = time.perf_counter()
    for fold in folds:
        train = data.loc[(data[DATETIME_COLUMN] >= fold.train_start) & (data[DATETIME_COLUMN] <= fold.train_end)]
        test = data.loc[(data[DATETIME_COLUMN] >= fold.test_start) & (data[DATETIME_COLUMN] <= fold.test_end)]
        for feature_set_name, features in feature_sets.items():
            model = build_multioutput_xgboost(features, args)
            model.fit(train[features], train[target_cols])
            predictions = pipeline_predict(model, test[features])
            if feature_set_name == "with_pollutant_context" and fold.fold == folds[-1].fold:
                fitted_for_explain = {
                    "model": model,
                    "features": features,
                    "test": test,
                    "groups": groups,
                    "targets": present_targets,
                    "feature_set": feature_set_name,
                }
            for target_idx, target in enumerate(present_targets):
                row = {
                    "target": target,
                    "horizon_h": horizon,
                    "fold": fold.fold,
                    "model": "multioutput_xgboost",
                    "feature_set": feature_set_name,
                    "n_targets": len(present_targets),
                    "n_features": len(features),
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                "embargo_hours": fold.embargo_hours,
                    "train_rows": len(train),
                    "test_rows": len(test),
                }
                row.update(metrics(test[target_cols[target_idx]], predictions[:, target_idx]))
                rows.append(row)
            progress.update(f"fold {fold.fold} {feature_set_name}")
    progress.close()
    timings = pd.DataFrame(
        [
            timing_row("multioutput", "prepare_feature_frame", prep_seconds, horizon=horizon),
            timing_row("multioutput", "prepare_feature_sets", setup_seconds, horizon=horizon),
            timing_row("multioutput", "cross_validation", time.perf_counter() - cv_start, horizon=horizon),
            timing_row("multioutput", "total", time.perf_counter() - total_start, horizon=horizon),
        ]
    )
    return pd.DataFrame(rows), fitted_for_explain, timings


def evaluate_single_feature_sets_xgboost(
    base_df: pd.DataFrame,
    target: str,
    horizon: int,
    args: argparse.Namespace,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not feature_sets:
        return pd.DataFrame(), pd.DataFrame()
    total_start = time.perf_counter()
    target_col = f"target_{target}"
    prep_start = time.perf_counter()
    data = advanced_feature_frame(base_df, target, horizon)
    data = data.dropna(subset=[target, target_col]).reset_index(drop=True)
    prep_seconds = time.perf_counter() - prep_start
    folds_start = time.perf_counter()
    folds = temporal_folds(data, args, horizon)
    folds_seconds = time.perf_counter() - folds_start
    rows: list[dict] = []
    progress = Progress(
        total=len(folds) * len(feature_sets),
        enabled=not args.no_progress,
        description=f"extended_xgboost {target} t+{horizon}h",
    )
    cv_start = time.perf_counter()
    for fold in folds:
        train = data.loc[(data[DATETIME_COLUMN] >= fold.train_start) & (data[DATETIME_COLUMN] <= fold.train_end)]
        test = data.loc[(data[DATETIME_COLUMN] >= fold.test_start) & (data[DATETIME_COLUMN] <= fold.test_end)]
        for feature_set_name, features in feature_sets.items():
            model = build_advanced_model("xgboost", features, args)
            model.fit(train[features], train[target_col])
            pred = pipeline_predict(model, test[features])
            row = {
                "target": target,
                "horizon_h": horizon,
                "fold": fold.fold,
                "model": "xgboost",
                "feature_set": feature_set_name,
                "n_features": len(features),
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "embargo_hours": fold.embargo_hours,
                "train_rows": len(train),
                "test_rows": len(test),
            }
            row.update(metrics(test[target_col], pred))
            rows.append(row)
            progress.update(f"fold {fold.fold} {feature_set_name}")
    progress.close()
    timings = pd.DataFrame(
        [
            timing_row("extended_single_ablation", "prepare_feature_frame", prep_seconds, target=target, horizon=horizon),
            timing_row("extended_single_ablation", "prepare_folds", folds_seconds, target=target, horizon=horizon),
            timing_row("extended_single_ablation", "cross_validation", time.perf_counter() - cv_start, target=target, horizon=horizon),
            timing_row("extended_single_ablation", "total", time.perf_counter() - total_start, target=target, horizon=horizon),
        ]
    )
    return pd.DataFrame(rows), timings


def evaluate_multioutput_feature_sets_xgboost(
    base_df: pd.DataFrame,
    targets: list[str],
    horizon: int,
    args: argparse.Namespace,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    present_targets = [target for target in targets if target in base_df.columns]
    if len(present_targets) < 2 or not feature_sets:
        return pd.DataFrame(), pd.DataFrame()
    total_start = time.perf_counter()
    target_cols = [target_output_column(target) for target in present_targets]
    prep_start = time.perf_counter()
    data = multioutput_feature_frame(base_df, present_targets, horizon)
    data = data.dropna(subset=target_cols).reset_index(drop=True)
    prep_seconds = time.perf_counter() - prep_start
    folds_start = time.perf_counter()
    folds = temporal_folds(data, args, horizon)
    folds_seconds = time.perf_counter() - folds_start
    rows: list[dict] = []
    progress = Progress(
        total=len(folds) * len(feature_sets),
        enabled=not args.no_progress,
        description=f"extended_multioutput_xgboost t+{horizon}h",
    )
    cv_start = time.perf_counter()
    for fold in folds:
        train = data.loc[(data[DATETIME_COLUMN] >= fold.train_start) & (data[DATETIME_COLUMN] <= fold.train_end)]
        test = data.loc[(data[DATETIME_COLUMN] >= fold.test_start) & (data[DATETIME_COLUMN] <= fold.test_end)]
        for feature_set_name, features in feature_sets.items():
            model = build_multioutput_xgboost(features, args)
            model.fit(train[features], train[target_cols])
            predictions = pipeline_predict(model, test[features])
            for target_idx, target in enumerate(present_targets):
                row = {
                    "target": target,
                    "horizon_h": horizon,
                    "fold": fold.fold,
                    "model": "multioutput_xgboost",
                    "feature_set": feature_set_name,
                    "n_targets": len(present_targets),
                    "n_features": len(features),
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                "embargo_hours": fold.embargo_hours,
                    "train_rows": len(train),
                    "test_rows": len(test),
                }
                row.update(metrics(test[target_cols[target_idx]], predictions[:, target_idx]))
                rows.append(row)
            progress.update(f"fold {fold.fold} {feature_set_name}")
    progress.close()
    timings = pd.DataFrame(
        [
            timing_row("extended_multi_ablation", "prepare_feature_frame", prep_seconds, horizon=horizon),
            timing_row("extended_multi_ablation", "prepare_folds", folds_seconds, horizon=horizon),
            timing_row("extended_multi_ablation", "cross_validation", time.perf_counter() - cv_start, horizon=horizon),
            timing_row("extended_multi_ablation", "total", time.perf_counter() - total_start, horizon=horizon),
        ]
    )
    return pd.DataFrame(rows), timings


def shap_by_group(fitted: dict, target: str, horizon: int, args: argparse.Namespace) -> pd.DataFrame:
    if args.no_shap or not fitted:
        return pd.DataFrame()
    shap_module = import_shap()
    if shap_module is None:
        return pd.DataFrame()
    model: Pipeline = fitted["model"]
    features: list[str] = fitted["features"]
    test: pd.DataFrame = fitted["test"]
    groups: dict[str, list[str]] = fitted["groups"]
    sample = test.sample(n=min(args.shap_sample, len(test)), random_state=42)
    transformed = pipeline_transform_features(model, sample[features])
    if "preprocess" in model.named_steps:
        feature_names = list(model.named_steps["preprocess"].get_feature_names_out())
    else:
        feature_names = list(features)
    explainer = shap_module.TreeExplainer(model.named_steps["model"])
    values = explainer.shap_values(transformed)
    abs_values = np.abs(values)
    rows: list[dict] = []
    for group_name, group_features in groups.items():
        present = [f for f in group_features if f in feature_names]
        if not present:
            continue
        idx = [feature_names.index(f) for f in present]
        rows.append(
            {
                "target": target,
                "horizon_h": horizon,
                "feature_set": fitted.get("feature_set", "with_target_autoregressive"),
                "group": group_name,
                "n_features": len(present),
                "mean_abs_shap": float(abs_values[:, idx].sum(axis=1).mean()),
            }
        )
    for idx, feature in enumerate(feature_names):
        rows.append(
            {
                "target": target,
                "horizon_h": horizon,
                "feature_set": fitted.get("feature_set", "with_target_autoregressive"),
                "group": "__feature__",
                "feature": feature,
                "n_features": 1,
                "mean_abs_shap": float(abs_values[:, idx].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)


def xgboost_native_importance(fitted: dict, target: str, horizon: int) -> pd.DataFrame:
    if not fitted:
        return pd.DataFrame()
    model: Pipeline = fitted["model"]
    if "model" not in model.named_steps:
        return pd.DataFrame()
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame()
    if "preprocess" in model.named_steps:
        feature_names = list(model.named_steps["preprocess"].get_feature_names_out())
    else:
        feature_names = list(fitted["features"])
    importances = np.asarray(estimator.feature_importances_, dtype=float)
    if importances.size != len(feature_names):
        return pd.DataFrame()
    total = float(importances.sum())
    normalized = importances / total if total > 0 else importances
    rows = [
        {
            "target": target,
            "horizon_h": horizon,
            "feature_set": fitted.get("feature_set", "with_target_autoregressive"),
            "feature": feature,
            "xgb_importance": float(importance),
        }
        for feature, importance in zip(feature_names, normalized)
    ]
    return pd.DataFrame(rows).sort_values("xgb_importance", ascending=False)


def multioutput_shap_by_group(fitted: dict, horizon: int, args: argparse.Namespace) -> pd.DataFrame:
    if args.no_shap or not fitted:
        return pd.DataFrame()
    shap_module = import_shap()
    if shap_module is None:
        return pd.DataFrame()
    model: Pipeline = fitted["model"]
    features: list[str] = fitted["features"]
    test: pd.DataFrame = fitted["test"]
    groups: dict[str, list[str]] = fitted["groups"]
    targets: list[str] = fitted["targets"]
    feature_set_name = fitted.get("feature_set", "with_pollutant_context")
    sample = test.sample(n=min(args.shap_sample, len(test)), random_state=42)
    transformed = pipeline_transform_features(model, sample[features])
    if "preprocess" in model.named_steps:
        feature_names = list(model.named_steps["preprocess"].get_feature_names_out())
    else:
        feature_names = list(features)
    rows: list[dict] = []
    multi_estimator = model.named_steps["model"]
    for target_idx, target in enumerate(targets):
        estimator = multi_estimator.estimators_[target_idx]
        explainer = shap_module.TreeExplainer(estimator)
        values = explainer.shap_values(transformed)
        abs_values = np.abs(values)
        for group_name, group_features in groups.items():
            present = [f for f in group_features if f in feature_names]
            if not present:
                continue
            idx = [feature_names.index(f) for f in present]
            rows.append(
                {
                    "target": target,
                    "horizon_h": horizon,
                    "feature_set": feature_set_name,
                    "group": group_name,
                    "n_features": len(present),
                    "mean_abs_shap": float(abs_values[:, idx].sum(axis=1).mean()),
                }
            )
        for idx, feature in enumerate(feature_names):
            rows.append(
                {
                    "target": target,
                    "horizon_h": horizon,
                    "feature_set": feature_set_name,
                    "group": "__feature__",
                    "feature": feature,
                    "n_features": 1,
                    "mean_abs_shap": float(abs_values[:, idx].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "mean_abs_shap"], ascending=[True, False])


def multioutput_xgboost_native_importance(fitted: dict, horizon: int) -> pd.DataFrame:
    if not fitted:
        return pd.DataFrame()
    model: Pipeline = fitted["model"]
    if "model" not in model.named_steps:
        return pd.DataFrame()
    multi_estimator = model.named_steps["model"]
    if not hasattr(multi_estimator, "estimators_"):
        return pd.DataFrame()
    if "preprocess" in model.named_steps:
        feature_names = list(model.named_steps["preprocess"].get_feature_names_out())
    else:
        feature_names = list(fitted["features"])
    rows: list[dict] = []
    feature_set_name = fitted.get("feature_set", "with_pollutant_context")
    for target_idx, target in enumerate(fitted["targets"]):
        estimator = multi_estimator.estimators_[target_idx]
        if not hasattr(estimator, "feature_importances_"):
            continue
        importances = np.asarray(estimator.feature_importances_, dtype=float)
        if importances.size != len(feature_names):
            continue
        total = float(importances.sum())
        normalized = importances / total if total > 0 else importances
        for feature, importance in zip(feature_names, normalized):
            rows.append(
                {
                    "target": target,
                    "horizon_h": horizon,
                    "feature_set": feature_set_name,
                    "feature": feature,
                    "xgb_importance": float(importance),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["target", "xgb_importance"], ascending=[True, False])


def build_xgboost_importance_summary(importances_out: pd.DataFrame) -> pd.DataFrame:
    return (
        importances_out.groupby(["target", "feature"], as_index=False)
        .agg(
            horizons=("horizon_h", "nunique"),
            mean_xgb_importance=("xgb_importance", "mean"),
            max_xgb_importance=("xgb_importance", "max"),
        )
        .sort_values(["target", "mean_xgb_importance"], ascending=[True, False])
    )


def build_multioutput_xgboost_importance_summary(importances_out: pd.DataFrame) -> pd.DataFrame:
    return (
        importances_out.groupby(["target", "feature_set", "feature"], as_index=False)
        .agg(
            horizons=("horizon_h", "nunique"),
            mean_xgb_importance=("xgb_importance", "mean"),
            max_xgb_importance=("xgb_importance", "max"),
        )
        .sort_values(["target", "feature_set", "mean_xgb_importance"], ascending=[True, True, False])
    )

def build_summary(scores_out: pd.DataFrame) -> pd.DataFrame:
    return (
        scores_out.groupby(["target", "horizon_h", "model", "feature_set"], as_index=False)
        .agg(
            folds=("fold", "count"),
            mean_MAE=("MAE", "mean"),
            std_MAE=("MAE", "std"),
            mean_RMSE=("RMSE", "mean"),
            mean_MAPE_pct=("MAPE_%", "mean"),
            mean_R2=("R2", "mean"),
            std_R2=("R2", "std"),
        )
        .sort_values(["target", "horizon_h", "feature_set", "mean_R2"], ascending=[True, True, True, False])
    )


def build_persistence_baseline_comparison(
    model_summary: pd.DataFrame, baseline_summary: pd.DataFrame
) -> pd.DataFrame:
    if model_summary.empty or baseline_summary.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    keys = ["target", "horizon_h"]
    model_best = model_summary.loc[model_summary.groupby(keys)["mean_R2"].idxmax()].copy()
    baseline_best = baseline_summary.loc[baseline_summary.groupby(keys)["mean_R2"].idxmax()].copy()
    merged = model_best.merge(
        baseline_best,
        on=keys,
        suffixes=("_best_model", "_best_baseline"),
    )
    for _, row in merged.iterrows():
        rows.append(
            {
                "target": row["target"],
                "horizon_h": row["horizon_h"],
                "best_model": row["model_best_model"],
                "best_model_feature_set": row["feature_set_best_model"],
                "best_model_R2": row["mean_R2_best_model"],
                "best_model_MAE": row["mean_MAE_best_model"],
                "best_baseline": row["model_best_baseline"],
                "best_baseline_R2": row["mean_R2_best_baseline"],
                "best_baseline_MAE": row["mean_MAE_best_baseline"],
                "delta_R2_vs_best_baseline": row["mean_R2_best_model"] - row["mean_R2_best_baseline"],
                "delta_MAE_vs_best_baseline": row["mean_MAE_best_model"] - row["mean_MAE_best_baseline"],
            }
        )
    return pd.DataFrame(rows).sort_values(["target", "horizon_h"])


def build_multioutput_summary(multi_scores_out: pd.DataFrame) -> pd.DataFrame:
    return (
        multi_scores_out.groupby(["target", "horizon_h", "model", "feature_set"], as_index=False)
        .agg(
            folds=("fold", "count"),
            n_targets=("n_targets", "max"),
            mean_MAE=("MAE", "mean"),
            std_MAE=("MAE", "std"),
            mean_RMSE=("RMSE", "mean"),
            mean_MAPE_pct=("MAPE_%", "mean"),
            mean_R2=("R2", "mean"),
            std_R2=("R2", "std"),
        )
        .sort_values(["target", "horizon_h", "feature_set", "mean_R2"], ascending=[True, True, True, False])
    )


def build_ablation_summary(summary_out: pd.DataFrame) -> pd.DataFrame:
    if summary_out.empty:
        return pd.DataFrame()
    comparisons = [
        ("no_target_autoregressive", "no_target_without_service_type", "service_type"),
        ("with_target_autoregressive", "with_target_without_service_type", "service_type"),
        ("no_target_autoregressive", "no_target_without_station_wind_bools", "station_wind_bools"),
        ("with_target_autoregressive", "with_target_without_station_wind_bools", "station_wind_bools"),
        (
            "no_target_autoregressive",
            "no_target_without_service_type_or_station_wind_bools",
            "service_type_and_station_wind_bools",
        ),
        (
            "with_target_autoregressive",
            "with_target_without_service_type_or_station_wind_bools",
            "service_type_and_station_wind_bools",
        ),
        ("no_pollutant_context", "no_pollutant_context_without_service_type", "service_type"),
        ("with_pollutant_context", "with_pollutant_context_without_service_type", "service_type"),
        ("no_pollutant_context", "no_pollutant_context_without_station_wind_bools", "station_wind_bools"),
        (
            "with_pollutant_context",
            "with_pollutant_context_without_station_wind_bools",
            "station_wind_bools",
        ),
        (
            "no_pollutant_context",
            "no_pollutant_context_without_service_type_or_station_wind_bools",
            "service_type_and_station_wind_bools",
        ),
        (
            "with_pollutant_context",
            "with_pollutant_context_without_service_type_or_station_wind_bools",
            "service_type_and_station_wind_bools",
        ),
    ]
    rows: list[dict] = []
    key_cols = ["target", "horizon_h", "model"]
    for full_set, ablated_set, removed_group in comparisons:
        full = summary_out.loc[summary_out["feature_set"] == full_set].copy()
        ablated = summary_out.loc[summary_out["feature_set"] == ablated_set].copy()
        if full.empty or ablated.empty:
            continue
        merged = full.merge(
            ablated,
            on=key_cols,
            suffixes=("_full", "_ablated"),
            how="inner",
        )
        for _, row in merged.iterrows():
            rows.append(
                {
                    "target": row["target"],
                    "horizon_h": row["horizon_h"],
                    "model": row["model"],
                    "full_feature_set": full_set,
                    "ablated_feature_set": ablated_set,
                    "removed_group": removed_group,
                    "full_mean_R2": row["mean_R2_full"],
                    "ablated_mean_R2": row["mean_R2_ablated"],
                    "r2_gain_when_included": row["mean_R2_full"] - row["mean_R2_ablated"],
                    "full_mean_MAE": row["mean_MAE_full"],
                    "ablated_mean_MAE": row["mean_MAE_ablated"],
                    "mae_reduction_when_included": row["mean_MAE_ablated"] - row["mean_MAE_full"],
                    "full_mean_RMSE": row["mean_RMSE_full"],
                    "ablated_mean_RMSE": row["mean_RMSE_ablated"],
                    "rmse_reduction_when_included": row["mean_RMSE_ablated"] - row["mean_RMSE_full"],
                    "full_mean_MAPE_pct": row["mean_MAPE_pct_full"],
                    "ablated_mean_MAPE_pct": row["mean_MAPE_pct_ablated"],
                    "mape_reduction_when_included": row["mean_MAPE_pct_ablated"] - row["mean_MAPE_pct_full"],
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["target", "horizon_h", "model", "removed_group"])


def bootstrap_ci(values: np.ndarray, repeats: int = EXTENDED_ABLATION_BOOTSTRAP_REPEATS) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan")
    if clean.size == 1:
        return float(clean[0]), float(clean[0])
    rng = np.random.default_rng(42)
    means = np.empty(repeats, dtype=float)
    for idx in range(repeats):
        sample = rng.choice(clean, size=clean.size, replace=True)
        means[idx] = sample.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_extended_feature_set_summary(
    scores_out: pd.DataFrame, metadata_df: pd.DataFrame, multioutput: bool = False
) -> pd.DataFrame:
    if scores_out.empty:
        return pd.DataFrame()
    summary = build_multioutput_summary(scores_out) if multioutput else build_summary(scores_out)
    if metadata_df.empty:
        return summary
    return summary.merge(metadata_df, on="feature_set", how="left").sort_values(
        ["target", "horizon_h", "feature_set", "mean_R2"], ascending=[True, True, True, False]
    )


def build_extended_delta_summary(
    scores_out: pd.DataFrame,
    summary_out: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scores_out.empty or summary_out.empty or metadata_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    fold_rows: list[dict] = []
    summary_rows: list[dict] = []

    for meta in metadata_df.to_dict("records"):
        if meta.get("analysis_type") in {"baseline", "only_block"}:
            continue
        feature_set = meta["feature_set"]
        previous_feature_set = meta.get("previous_feature_set")
        baseline_feature_set = meta.get("baseline_feature_set")
        if pd.isna(previous_feature_set):
            previous_feature_set = None
        if pd.isna(baseline_feature_set):
            baseline_feature_set = None
        compare_to = previous_feature_set or baseline_feature_set
        if not compare_to:
            continue

        subset = scores_out.loc[scores_out["feature_set"] == feature_set, ["target", "horizon_h", "model"]].drop_duplicates()
        for _, key_row in subset.iterrows():
            key = (key_row["target"], key_row["horizon_h"], key_row["model"])
            full = summary_out.loc[
                (summary_out["target"] == key[0])
                & (summary_out["horizon_h"] == key[1])
                & (summary_out["model"] == key[2])
                & (summary_out["feature_set"] == compare_to)
            ]
            ablated = summary_out.loc[
                (summary_out["target"] == key[0])
                & (summary_out["horizon_h"] == key[1])
                & (summary_out["model"] == key[2])
                & (summary_out["feature_set"] == feature_set)
            ]
            if full.empty or ablated.empty:
                continue
            full = full.iloc[0]
            ablated = ablated.iloc[0]
            fold_subset = scores_out.loc[
                (scores_out["target"] == key[0])
                & (scores_out["horizon_h"] == key[1])
                & (scores_out["model"] == key[2])
                & (scores_out["feature_set"].isin([feature_set, compare_to]))
            ][["fold", "feature_set", "R2", "MAE", "RMSE", "MAPE_%"]]
            pivot = fold_subset.pivot(index="fold", columns="feature_set", values=["R2", "MAE", "RMSE", "MAPE_%"])
            if (("R2", feature_set) not in pivot.columns) or (("R2", compare_to) not in pivot.columns):
                continue
            delta_r2 = pivot[("R2", compare_to)] - pivot[("R2", feature_set)]
            delta_mae = pivot[("MAE", feature_set)] - pivot[("MAE", compare_to)]
            delta_rmse = pivot[("RMSE", feature_set)] - pivot[("RMSE", compare_to)]
            delta_mape = pivot[("MAPE_%", feature_set)] - pivot[("MAPE_%", compare_to)]

            for fold in delta_r2.index:
                fold_rows.append(
                    {
                        **meta,
                        "target": key[0],
                        "horizon_h": key[1],
                        "model": key[2],
                        "reference_feature_set": compare_to,
                        "candidate_feature_set": feature_set,
                        "fold": int(fold),
                        "delta_R2": float(delta_r2.loc[fold]),
                        "delta_MAE_reduction": float(delta_mae.loc[fold]),
                        "delta_RMSE_reduction": float(delta_rmse.loc[fold]),
                        "delta_MAPE_reduction": float(delta_mape.loc[fold]),
                    }
                )

            r2_ci_low, r2_ci_high = bootstrap_ci(delta_r2.to_numpy())
            mae_ci_low, mae_ci_high = bootstrap_ci(delta_mae.to_numpy())
            rmse_ci_low, rmse_ci_high = bootstrap_ci(delta_rmse.to_numpy())
            mape_ci_low, mape_ci_high = bootstrap_ci(delta_mape.to_numpy())
            summary_rows.append(
                {
                    **meta,
                    "target": key[0],
                    "horizon_h": key[1],
                    "model": key[2],
                    "reference_feature_set": compare_to,
                    "candidate_feature_set": feature_set,
                    "reference_mean_R2": float(full["mean_R2"]),
                    "candidate_mean_R2": float(ablated["mean_R2"]),
                    "delta_R2_mean": float(delta_r2.mean()),
                    "delta_R2_std": float(delta_r2.std(ddof=0)),
                    "delta_R2_ci95_low": r2_ci_low,
                    "delta_R2_ci95_high": r2_ci_high,
                    "delta_MAE_reduction_mean": float(delta_mae.mean()),
                    "delta_MAE_reduction_std": float(delta_mae.std(ddof=0)),
                    "delta_MAE_ci95_low": mae_ci_low,
                    "delta_MAE_ci95_high": mae_ci_high,
                    "delta_RMSE_reduction_mean": float(delta_rmse.mean()),
                    "delta_RMSE_reduction_std": float(delta_rmse.std(ddof=0)),
                    "delta_RMSE_ci95_low": rmse_ci_low,
                    "delta_RMSE_ci95_high": rmse_ci_high,
                    "delta_MAPE_reduction_mean": float(delta_mape.mean()),
                    "delta_MAPE_reduction_std": float(delta_mape.std(ddof=0)),
                    "delta_MAPE_ci95_low": mape_ci_low,
                    "delta_MAPE_ci95_high": mape_ci_high,
                    "n_folds_compared": int(delta_r2.notna().sum()),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not fold_df.empty:
        fold_df = fold_df.sort_values(["target", "horizon_h", "model", "analysis_type", "candidate_feature_set", "fold"])
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["target", "horizon_h", "model", "analysis_type", "candidate_feature_set"])
    return fold_df, summary_df


def save_best_model_metric_plots(summary_out: pd.DataFrame, output_dir: Path) -> None:
    if summary_out.empty:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_info = [
        ("mean_MAE", "MAE"),
        ("mean_R2", "R2"),
        ("mean_RMSE", "RMSE"),
        ("mean_MAPE_pct", "MAPE (%)"),
    ]
    for target in sorted(summary_out["target"].unique()):
        target_rows = summary_out.loc[summary_out["target"] == target].copy()
        at_1h = target_rows.loc[target_rows["horizon_h"] == 1].sort_values("mean_R2", ascending=False)
        if at_1h.empty:
            continue
        best = at_1h.iloc[0]
        selected = target_rows.loc[
            (target_rows["model"] == best["model"]) & (target_rows["feature_set"] == best["feature_set"])
        ].sort_values("horizon_h")
        if selected.empty:
            continue

        horizons = selected["horizon_h"].to_numpy()
        feature_set_label = str(best["feature_set"]).replace("_", " ")
        for metric_col, metric_label in metrics_info:
            fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
            values = selected[metric_col].to_numpy()
            ax.plot(horizons, values, marker="o", linewidth=2, label=metric_label)
            ax.set_xlabel("Orizzonte (h)")
            ax.set_ylabel(metric_label)
            ax.set_xticks(horizons)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            ax.set_title(
                f"{target} - {metric_label} del best @1h: {best['model']} | {feature_set_label} | R2={best['mean_R2']:.3f}"
            )
            metric_slug = metric_label.lower().replace(" (%)", "_pct").replace(" ", "_")
            out_path = plots_dir / f"{target}_best_model_{metric_slug}.{PLOT_FORMAT}"
            fig.savefig(out_path, bbox_inches="tight", format=PLOT_FORMAT)
            plt.close(fig)


def save_model_test_series_plots(
    predictions_out: pd.DataFrame, summary_out: pd.DataFrame, output_dir: Path
) -> None:
    if predictions_out.empty or summary_out.empty:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    target_summaries = summary_out.loc[summary_out["horizon_h"] == 1].copy()
    if target_summaries.empty:
        return

    for target in sorted(target_summaries["target"].unique()):
        target_pred = predictions_out.loc[
            (predictions_out["target"] == target) & (predictions_out["horizon_h"] == 1)
        ].copy()
        if target_pred.empty:
            continue
        target_best = (
            target_summaries.loc[target_summaries["target"] == target]
            .sort_values(["model", "mean_R2"], ascending=[True, False])
            .drop_duplicates(subset=["model"], keep="first")
        )
        if target_best.empty:
            continue

        for fold in sorted(target_pred["fold"].unique()):
            fold_pred = target_pred.loc[target_pred["fold"] == fold].copy()
            if fold_pred.empty:
                continue
            actual = (
                fold_pred[["datetime", "y_true"]]
                .groupby("datetime", as_index=False)["y_true"]
                .mean()
                .sort_values("datetime")
            )
            fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
            ax.plot(actual["datetime"], actual["y_true"], color="black", linewidth=2, label="target reale")
            for _, best in target_best.iterrows():
                series = fold_pred.loc[
                    (fold_pred["model"] == best["model"]) & (fold_pred["feature_set"] == best["feature_set"])
                ]
                if series.empty:
                    continue
                series = (
                    series[["datetime", "y_pred"]]
                    .groupby("datetime", as_index=False)["y_pred"]
                    .mean()
                    .sort_values("datetime")
                )
                label = f"{best['model']} ({str(best['feature_set']).replace('_', ' ')})"
                ax.plot(series["datetime"], series["y_pred"], linewidth=1.5, label=label)
            ax.set_title(f"{target} - Predizioni sul test set t+1h - fold {fold}")
            ax.set_xlabel("Tempo")
            ax.set_ylabel(target)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
            out_path = plots_dir / f"{target}_model_test_series_1h_fold_{fold}.{PLOT_FORMAT}"
            fig.savefig(out_path, bbox_inches="tight", format=PLOT_FORMAT)
            plt.close(fig)


_GLOBAL_DF: pd.DataFrame | None = None
_GLOBAL_ARGS: argparse.Namespace | None = None


def _init_worker(df: pd.DataFrame, args: argparse.Namespace) -> None:
    global _GLOBAL_DF, _GLOBAL_ARGS
    _GLOBAL_DF = df
    _GLOBAL_ARGS = args


def _run_single_task(task: tuple[str, int]) -> tuple[str, int, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target, horizon = task
    scores, predictions, fitted, timings = evaluate_temporal_cv(_GLOBAL_DF, target, horizon, _GLOBAL_ARGS)
    shap_start = time.perf_counter()
    shap_df = shap_by_group(fitted, target, horizon, _GLOBAL_ARGS)
    shap_seconds = time.perf_counter() - shap_start
    native_start = time.perf_counter()
    xgb_native_df = xgboost_native_importance(fitted, target, horizon)
    native_seconds = time.perf_counter() - native_start
    extra = pd.DataFrame(
        [
            timing_row("single_target", "shap", shap_seconds, target=target, horizon=horizon),
            timing_row("single_target", "xgb_native_importance", native_seconds, target=target, horizon=horizon),
        ]
    )
    return target, horizon, scores, predictions, shap_df, xgb_native_df, pd.concat([timings, extra], ignore_index=True)


def _run_multi_task(horizon: int) -> tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    multi_scores, fitted, timings = evaluate_multioutput_xgboost(_GLOBAL_DF, _GLOBAL_ARGS.targets, horizon, _GLOBAL_ARGS)
    shap_start = time.perf_counter()
    multi_shap = multioutput_shap_by_group(fitted, horizon, _GLOBAL_ARGS)
    shap_seconds = time.perf_counter() - shap_start
    native_start = time.perf_counter()
    multi_importance = multioutput_xgboost_native_importance(fitted, horizon)
    native_seconds = time.perf_counter() - native_start
    extra = pd.DataFrame(
        [
            timing_row("multioutput", "shap", shap_seconds, horizon=horizon),
            timing_row("multioutput", "xgb_native_importance", native_seconds, horizon=horizon),
        ]
    )
    return horizon, multi_scores, multi_shap, multi_importance, pd.concat([timings, extra], ignore_index=True)


def _run_extended_single_ablation_task(task: tuple[str, int]) -> tuple[str, int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target, horizon = task
    setup_start = time.perf_counter()
    feature_frame = advanced_feature_frame(_GLOBAL_DF, target, horizon)
    target_col = f"target_{target}"
    feature_candidates = [c for c in feature_frame.columns if c not in {DATETIME_COLUMN, target_col}]
    groups = advanced_feature_groups(feature_candidates, target)
    feature_sets, metadata = build_single_extended_ablation_feature_sets(groups)
    setup_seconds = time.perf_counter() - setup_start
    scores, timings = evaluate_single_feature_sets_xgboost(_GLOBAL_DF, target, horizon, _GLOBAL_ARGS, feature_sets)
    extra = pd.DataFrame([timing_row("extended_single_ablation", "prepare_feature_sets", setup_seconds, target=target, horizon=horizon)])
    return target, horizon, scores, metadata, pd.concat([extra, timings], ignore_index=True)


def _run_extended_multi_ablation_task(horizon: int) -> tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    present_targets = [target for target in _GLOBAL_ARGS.targets if target in _GLOBAL_DF.columns]
    setup_start = time.perf_counter()
    feature_frame = multioutput_feature_frame(_GLOBAL_DF, present_targets, horizon)
    target_cols = [target_output_column(target) for target in present_targets]
    feature_candidates = [c for c in feature_frame.columns if c not in {DATETIME_COLUMN, *target_cols}]
    groups = multioutput_feature_groups(feature_candidates, present_targets)
    feature_sets, metadata = build_multioutput_extended_ablation_feature_sets(groups)
    setup_seconds = time.perf_counter() - setup_start
    scores, timings = evaluate_multioutput_feature_sets_xgboost(_GLOBAL_DF, present_targets, horizon, _GLOBAL_ARGS, feature_sets)
    extra = pd.DataFrame([timing_row("extended_multi_ablation", "prepare_feature_sets", setup_seconds, horizon=horizon)])
    return horizon, scores, metadata, pd.concat([extra, timings], ignore_index=True)


def run_analysis(args: argparse.Namespace) -> int:
    overall_start = time.perf_counter()
    df = read_dataset(args.input)
    runtime_rows: list[dict] = [timing_row("main", "read_dataset", time.perf_counter() - overall_start)]
    base_start = time.perf_counter()
    advanced_base_df = build_advanced_base_frame(df, args)
    runtime_rows.append(timing_row("main", "build_advanced_base_frame", time.perf_counter() - base_start))
    all_scores: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_baseline_scores: list[pd.DataFrame] = []
    all_baseline_predictions: list[pd.DataFrame] = []
    all_shap: list[pd.DataFrame] = []
    all_xgb_native_importance: list[pd.DataFrame] = []
    all_multioutput_scores: list[pd.DataFrame] = []
    all_multioutput_shap: list[pd.DataFrame] = []
    all_multioutput_xgb_native_importance: list[pd.DataFrame] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "advanced_temporal_cv_scores.csv"
    predictions_path = args.output_dir / "advanced_temporal_cv_predictions.csv"
    summary_path = args.output_dir / "advanced_temporal_cv_summary.csv"
    baseline_scores_path = args.output_dir / "advanced_persistence_baseline_scores.csv"
    baseline_predictions_path = args.output_dir / "advanced_persistence_baseline_predictions.csv"
    baseline_summary_path = args.output_dir / "advanced_persistence_baseline_summary.csv"
    baseline_comparison_path = args.output_dir / "advanced_persistence_baseline_comparison.csv"
    feature_inventory_path = args.output_dir / "advanced_feature_inventory.csv"
    feature_group_inventory_path = args.output_dir / "advanced_feature_group_inventory.csv"
    validation_folds_path = args.output_dir / "advanced_validation_folds.csv"
    pollutant_stats_path = args.output_dir / "pollutant_station_reference_stats.csv"
    ablation_summary_path = args.output_dir / "advanced_ablation_summary.csv"
    shap_path = args.output_dir / "advanced_group_shap.csv"
    xgb_native_path = args.output_dir / "advanced_xgboost_native_feature_importances.csv"
    xgb_native_summary_path = args.output_dir / "advanced_xgboost_native_feature_importances_summary.csv"
    multi_scores_path = args.output_dir / "advanced_multioutput_xgboost_scores.csv"
    multi_summary_path = args.output_dir / "advanced_multioutput_xgboost_summary.csv"
    multi_ablation_summary_path = args.output_dir / "advanced_multioutput_ablation_summary.csv"
    multi_shap_path = args.output_dir / "advanced_multioutput_group_shap.csv"
    multi_xgb_native_path = args.output_dir / "advanced_multioutput_xgboost_native_feature_importances.csv"
    multi_xgb_native_summary_path = (
        args.output_dir / "advanced_multioutput_xgboost_native_feature_importances_summary.csv"
    )
    extended_single_feature_sets_path = args.output_dir / "advanced_extended_ablation_feature_sets.csv"
    extended_single_scores_path = args.output_dir / "advanced_extended_ablation_scores.csv"
    extended_single_summary_path = args.output_dir / "advanced_extended_ablation_summary.csv"
    extended_single_fold_deltas_path = args.output_dir / "advanced_extended_ablation_fold_deltas.csv"
    extended_single_delta_summary_path = args.output_dir / "advanced_extended_ablation_delta_summary.csv"
    extended_multi_feature_sets_path = args.output_dir / "advanced_multioutput_extended_ablation_feature_sets.csv"
    extended_multi_scores_path = args.output_dir / "advanced_multioutput_extended_ablation_scores.csv"
    extended_multi_summary_path = args.output_dir / "advanced_multioutput_extended_ablation_summary.csv"
    extended_multi_fold_deltas_path = args.output_dir / "advanced_multioutput_extended_ablation_fold_deltas.csv"
    extended_multi_delta_summary_path = args.output_dir / "advanced_multioutput_extended_ablation_delta_summary.csv"
    runtime_profile_path = args.output_dir / "advanced_runtime_profile.csv"

    single_tasks = [(target, horizon) for target in args.targets for horizon in args.horizons]
    single_workers = resolve_workers(args.workers, len(single_tasks))
    worker_args = argparse.Namespace(**vars(args))
    worker_args.no_progress = True
    stats_start = time.perf_counter()
    pollutant_stats = build_pollutant_station_stats(df)
    pollutant_stats.to_csv(pollutant_stats_path, index=False)
    runtime_rows.append(timing_row("main", "pollutant_stats", time.perf_counter() - stats_start))
    inventory_start = time.perf_counter()
    feature_inventory, feature_group_inventory, validation_folds = build_feature_inventory(
        advanced_base_df, args.targets, args.horizons, args
    )
    if not feature_inventory.empty:
        feature_inventory.to_csv(feature_inventory_path, index=False)
    if not feature_group_inventory.empty:
        feature_group_inventory.to_csv(feature_group_inventory_path, index=False)
    if not validation_folds.empty:
        validation_folds.to_csv(validation_folds_path, index=False)
    runtime_rows.append(timing_row("main", "feature_inventory", time.perf_counter() - inventory_start))
    print(f"Worker single-target: {single_workers}")

    if single_workers == 1:
        for target, horizon in single_tasks:
            baseline_scores, baseline_predictions = evaluate_persistence_baselines(
                advanced_base_df, target, horizon, args
            )
            if not baseline_scores.empty:
                all_baseline_scores.append(baseline_scores)
                baseline_scores_out = pd.concat(all_baseline_scores, ignore_index=True)
                baseline_scores_out.to_csv(baseline_scores_path, index=False)
                build_summary(baseline_scores_out).to_csv(baseline_summary_path, index=False)
            if not baseline_predictions.empty:
                all_baseline_predictions.append(baseline_predictions)
                pd.concat(all_baseline_predictions, ignore_index=True).to_csv(baseline_predictions_path, index=False)
            scores, predictions, fitted, timings = evaluate_temporal_cv(advanced_base_df, target, horizon, args)
            runtime_rows.extend(timings.to_dict("records"))
            shap_start = time.perf_counter()
            shap_df = shap_by_group(fitted, target, horizon, args)
            runtime_rows.append(timing_row("single_target", "shap", time.perf_counter() - shap_start, target=target, horizon=horizon))
            native_start = time.perf_counter()
            xgb_native_df = xgboost_native_importance(fitted, target, horizon)
            runtime_rows.append(timing_row("single_target", "xgb_native_importance", time.perf_counter() - native_start, target=target, horizon=horizon))
            all_scores.append(scores)
            all_predictions.append(predictions)
            if not shap_df.empty:
                all_shap.append(shap_df)
            if not xgb_native_df.empty:
                all_xgb_native_importance.append(xgb_native_df)
            scores_out = pd.concat(all_scores, ignore_index=True)
            predictions_out = pd.concat(all_predictions, ignore_index=True)
            scores_out.to_csv(scores_path, index=False)
            predictions_out.to_csv(predictions_path, index=False)
            summary_out = build_summary(scores_out)
            summary_out.to_csv(summary_path, index=False)
            if all_baseline_scores:
                baseline_summary_out = build_summary(pd.concat(all_baseline_scores, ignore_index=True))
                build_persistence_baseline_comparison(summary_out, baseline_summary_out).to_csv(
                    baseline_comparison_path, index=False
                )
            build_ablation_summary(summary_out).to_csv(ablation_summary_path, index=False)
            save_best_model_metric_plots(summary_out, args.output_dir)
            save_model_test_series_plots(predictions_out, summary_out, args.output_dir)
            if all_shap:
                pd.concat(all_shap, ignore_index=True).to_csv(shap_path, index=False)
            if all_xgb_native_importance:
                native_out = pd.concat(all_xgb_native_importance, ignore_index=True)
                native_out.to_csv(xgb_native_path, index=False)
                build_xgboost_importance_summary(native_out).to_csv(xgb_native_summary_path, index=False)
            print(f"Checkpoint scritto: {target} t+{horizon}h")
    else:
        with ProcessPoolExecutor(
            max_workers=single_workers,
            initializer=_init_worker,
            initargs=(advanced_base_df, worker_args),
        ) as executor:
            futures = {executor.submit(_run_single_task, task): task for task in single_tasks}
            for future in as_completed(futures):
                target, horizon, scores, predictions, shap_df, xgb_native_df, timings = future.result()
                baseline_scores, baseline_predictions = evaluate_persistence_baselines(
                    advanced_base_df, target, horizon, args
                )
                if not baseline_scores.empty:
                    all_baseline_scores.append(baseline_scores)
                    baseline_scores_out = pd.concat(all_baseline_scores, ignore_index=True)
                    baseline_scores_out.to_csv(baseline_scores_path, index=False)
                    build_summary(baseline_scores_out).to_csv(baseline_summary_path, index=False)
                if not baseline_predictions.empty:
                    all_baseline_predictions.append(baseline_predictions)
                    pd.concat(all_baseline_predictions, ignore_index=True).to_csv(baseline_predictions_path, index=False)
                runtime_rows.extend(timings.to_dict("records"))
                all_scores.append(scores)
                all_predictions.append(predictions)
                if not shap_df.empty:
                    all_shap.append(shap_df)
                if not xgb_native_df.empty:
                    all_xgb_native_importance.append(xgb_native_df)
                scores_out = pd.concat(all_scores, ignore_index=True)
                predictions_out = pd.concat(all_predictions, ignore_index=True)
                scores_out.to_csv(scores_path, index=False)
                predictions_out.to_csv(predictions_path, index=False)
                summary_out = build_summary(scores_out)
                summary_out.to_csv(summary_path, index=False)
                if all_baseline_scores:
                    baseline_summary_out = build_summary(pd.concat(all_baseline_scores, ignore_index=True))
                    build_persistence_baseline_comparison(summary_out, baseline_summary_out).to_csv(
                        baseline_comparison_path, index=False
                    )
                build_ablation_summary(summary_out).to_csv(ablation_summary_path, index=False)
                save_best_model_metric_plots(summary_out, args.output_dir)
                save_model_test_series_plots(predictions_out, summary_out, args.output_dir)
                if all_shap:
                    pd.concat(all_shap, ignore_index=True).to_csv(shap_path, index=False)
                if all_xgb_native_importance:
                    native_out = pd.concat(all_xgb_native_importance, ignore_index=True)
                    native_out.to_csv(xgb_native_path, index=False)
                    build_xgboost_importance_summary(native_out).to_csv(xgb_native_summary_path, index=False)
                print(f"Checkpoint scritto: {target} t+{horizon}h")

    extended_single_scores: list[pd.DataFrame] = []
    extended_single_metadata_frames: list[pd.DataFrame] = []
    if single_workers == 1:
        for target, horizon in single_tasks:
            setup_start = time.perf_counter()
            feature_frame = advanced_feature_frame(advanced_base_df, target, horizon)
            target_col = f"target_{target}"
            feature_candidates = [c for c in feature_frame.columns if c not in {DATETIME_COLUMN, target_col}]
            groups = advanced_feature_groups(feature_candidates, target)
            feature_sets, metadata = build_single_extended_ablation_feature_sets(groups)
            runtime_rows.append(timing_row("extended_single_ablation", "prepare_feature_sets", time.perf_counter() - setup_start, target=target, horizon=horizon))
            scores, timings = evaluate_single_feature_sets_xgboost(advanced_base_df, target, horizon, args, feature_sets)
            runtime_rows.extend(timings.to_dict("records"))
            if not scores.empty:
                extended_single_scores.append(scores)
            if not metadata.empty:
                extended_single_metadata_frames.append(metadata)
    else:
        with ProcessPoolExecutor(
            max_workers=single_workers,
            initializer=_init_worker,
            initargs=(advanced_base_df, worker_args),
        ) as executor:
            futures = {executor.submit(_run_extended_single_ablation_task, task): task for task in single_tasks}
            for future in as_completed(futures):
                _, _, scores, metadata, timings = future.result()
                runtime_rows.extend(timings.to_dict("records"))
                if not scores.empty:
                    extended_single_scores.append(scores)
                if not metadata.empty:
                    extended_single_metadata_frames.append(metadata)

    extended_single_scores_out = pd.DataFrame()
    extended_single_summary_out = pd.DataFrame()
    extended_single_delta_summary = pd.DataFrame()
    if extended_single_scores:
        extended_single_scores_out = pd.concat(extended_single_scores, ignore_index=True)
        extended_single_metadata = (
            pd.concat(extended_single_metadata_frames, ignore_index=True).drop_duplicates(subset=["feature_set"])
            if extended_single_metadata_frames
            else pd.DataFrame()
        )
        extended_single_scores_out.to_csv(extended_single_scores_path, index=False)
        if not extended_single_metadata.empty:
            extended_single_metadata.to_csv(extended_single_feature_sets_path, index=False)
        extended_single_summary_out = build_extended_feature_set_summary(
            extended_single_scores_out, extended_single_metadata
        )
        extended_single_summary_out.to_csv(extended_single_summary_path, index=False)
        extended_single_fold_deltas, extended_single_delta_summary = build_extended_delta_summary(
            extended_single_scores_out, extended_single_summary_out, extended_single_metadata
        )
        if not extended_single_fold_deltas.empty:
            extended_single_fold_deltas.to_csv(extended_single_fold_deltas_path, index=False)
        if not extended_single_delta_summary.empty:
            extended_single_delta_summary.to_csv(extended_single_delta_summary_path, index=False)

    if not args.no_multioutput_xgb:
        multi_workers = resolve_workers(args.workers, len(args.horizons))
        print(f"Worker multioutput: {multi_workers}")
        if multi_workers == 1:
            for horizon in args.horizons:
                multi_scores, fitted, timings = evaluate_multioutput_xgboost(advanced_base_df, args.targets, horizon, args)
                runtime_rows.extend(timings.to_dict("records"))
                shap_start = time.perf_counter()
                multi_shap = multioutput_shap_by_group(fitted, horizon, args)
                runtime_rows.append(timing_row("multioutput", "shap", time.perf_counter() - shap_start, horizon=horizon))
                native_start = time.perf_counter()
                multi_importance = multioutput_xgboost_native_importance(fitted, horizon)
                runtime_rows.append(timing_row("multioutput", "xgb_native_importance", time.perf_counter() - native_start, horizon=horizon))
                if not multi_scores.empty:
                    all_multioutput_scores.append(multi_scores)
                    multi_scores_out = pd.concat(all_multioutput_scores, ignore_index=True)
                    multi_scores_out.to_csv(multi_scores_path, index=False)
                    multi_summary_out = build_multioutput_summary(multi_scores_out)
                    multi_summary_out.to_csv(multi_summary_path, index=False)
                    build_ablation_summary(multi_summary_out).to_csv(multi_ablation_summary_path, index=False)
                if not multi_shap.empty:
                    all_multioutput_shap.append(multi_shap)
                    pd.concat(all_multioutput_shap, ignore_index=True).to_csv(multi_shap_path, index=False)
                if not multi_importance.empty:
                    all_multioutput_xgb_native_importance.append(multi_importance)
                    multi_native_out = pd.concat(all_multioutput_xgb_native_importance, ignore_index=True)
                    multi_native_out.to_csv(multi_xgb_native_path, index=False)
                    build_multioutput_xgboost_importance_summary(multi_native_out).to_csv(
                        multi_xgb_native_summary_path, index=False
                    )
                print(f"Checkpoint multioutput scritto: t+{horizon}h")
        else:
            with ProcessPoolExecutor(
                max_workers=multi_workers,
                initializer=_init_worker,
                initargs=(advanced_base_df, worker_args),
            ) as executor:
                futures = {executor.submit(_run_multi_task, horizon): horizon for horizon in args.horizons}
                for future in as_completed(futures):
                    horizon, multi_scores, multi_shap, multi_importance, timings = future.result()
                    runtime_rows.extend(timings.to_dict("records"))
                    if not multi_scores.empty:
                        all_multioutput_scores.append(multi_scores)
                        multi_scores_out = pd.concat(all_multioutput_scores, ignore_index=True)
                        multi_scores_out.to_csv(multi_scores_path, index=False)
                        multi_summary_out = build_multioutput_summary(multi_scores_out)
                        multi_summary_out.to_csv(multi_summary_path, index=False)
                        build_ablation_summary(multi_summary_out).to_csv(multi_ablation_summary_path, index=False)
                    if not multi_shap.empty:
                        all_multioutput_shap.append(multi_shap)
                        pd.concat(all_multioutput_shap, ignore_index=True).to_csv(multi_shap_path, index=False)
                    if not multi_importance.empty:
                        all_multioutput_xgb_native_importance.append(multi_importance)
                        multi_native_out = pd.concat(all_multioutput_xgb_native_importance, ignore_index=True)
                        multi_native_out.to_csv(multi_xgb_native_path, index=False)
                        build_multioutput_xgboost_importance_summary(multi_native_out).to_csv(
                            multi_xgb_native_summary_path, index=False
                        )
                    print(f"Checkpoint multioutput scritto: t+{horizon}h")

        extended_multi_scores: list[pd.DataFrame] = []
        extended_multi_metadata_frames: list[pd.DataFrame] = []
        if multi_workers == 1:
            for horizon in args.horizons:
                present_targets = [target for target in args.targets if target in df.columns]
                setup_start = time.perf_counter()
                feature_frame = multioutput_feature_frame(advanced_base_df, present_targets, horizon)
                target_cols = [target_output_column(target) for target in present_targets]
                feature_candidates = [c for c in feature_frame.columns if c not in {DATETIME_COLUMN, *target_cols}]
                groups = multioutput_feature_groups(feature_candidates, present_targets)
                feature_sets, metadata = build_multioutput_extended_ablation_feature_sets(groups)
                runtime_rows.append(timing_row("extended_multi_ablation", "prepare_feature_sets", time.perf_counter() - setup_start, horizon=horizon))
                scores, timings = evaluate_multioutput_feature_sets_xgboost(advanced_base_df, present_targets, horizon, args, feature_sets)
                runtime_rows.extend(timings.to_dict("records"))
                if not scores.empty:
                    extended_multi_scores.append(scores)
                if not metadata.empty:
                    extended_multi_metadata_frames.append(metadata)
        else:
            with ProcessPoolExecutor(
                max_workers=multi_workers,
                initializer=_init_worker,
                initargs=(advanced_base_df, worker_args),
            ) as executor:
                futures = {executor.submit(_run_extended_multi_ablation_task, horizon): horizon for horizon in args.horizons}
                for future in as_completed(futures):
                    _, scores, metadata, timings = future.result()
                    runtime_rows.extend(timings.to_dict("records"))
                    if not scores.empty:
                        extended_multi_scores.append(scores)
                    if not metadata.empty:
                        extended_multi_metadata_frames.append(metadata)

        extended_multi_scores_out = pd.DataFrame()
        extended_multi_summary_out = pd.DataFrame()
        extended_multi_delta_summary = pd.DataFrame()
        if extended_multi_scores:
            extended_multi_scores_out = pd.concat(extended_multi_scores, ignore_index=True)
            extended_multi_metadata = (
                pd.concat(extended_multi_metadata_frames, ignore_index=True).drop_duplicates(subset=["feature_set"])
                if extended_multi_metadata_frames
                else pd.DataFrame()
            )
            extended_multi_scores_out.to_csv(extended_multi_scores_path, index=False)
            if not extended_multi_metadata.empty:
                extended_multi_metadata.to_csv(extended_multi_feature_sets_path, index=False)
            extended_multi_summary_out = build_extended_feature_set_summary(
                extended_multi_scores_out, extended_multi_metadata, multioutput=True
            )
            extended_multi_summary_out.to_csv(extended_multi_summary_path, index=False)
            extended_multi_fold_deltas, extended_multi_delta_summary = build_extended_delta_summary(
                extended_multi_scores_out, extended_multi_summary_out, extended_multi_metadata
            )
            if not extended_multi_fold_deltas.empty:
                extended_multi_fold_deltas.to_csv(extended_multi_fold_deltas_path, index=False)
            if not extended_multi_delta_summary.empty:
                extended_multi_delta_summary.to_csv(extended_multi_delta_summary_path, index=False)

    runtime_rows.append(timing_row("main", "total", time.perf_counter() - overall_start))
    if runtime_rows:
        pd.DataFrame(runtime_rows).to_csv(runtime_profile_path, index=False)

    print(f"File scritto: {scores_path}")
    print(f"File scritto: {predictions_path}")
    print(f"File scritto: {summary_path}")
    if all_baseline_scores:
        print(f"File scritto: {baseline_scores_path}")
        print(f"File scritto: {baseline_summary_path}")
        print(f"File scritto: {baseline_comparison_path}")
    if not feature_inventory.empty:
        print(f"File scritto: {feature_inventory_path}")
    if not feature_group_inventory.empty:
        print(f"File scritto: {feature_group_inventory_path}")
    if not validation_folds.empty:
        print(f"File scritto: {validation_folds_path}")
    print(f"File scritto: {pollutant_stats_path}")
    print(f"File scritto: {ablation_summary_path}")
    if (args.output_dir / "plots").exists():
        print(f"Cartella grafici: {args.output_dir / 'plots'}")
    if all_multioutput_scores:
        print(f"File scritto: {multi_scores_path}")
        print(f"File scritto: {multi_summary_path}")
        print(f"File scritto: {multi_ablation_summary_path}")
    if all_multioutput_shap:
        print(f"File scritto: {multi_shap_path}")
    if all_multioutput_xgb_native_importance:
        print(f"File scritto: {multi_xgb_native_path}")
        print(f"File scritto: {multi_xgb_native_summary_path}")
    if all_shap:
        print(f"File scritto: {shap_path}")
    elif not args.no_shap:
        print("SHAP non prodotto: richiede libreria shap, modello xgboost e feature set autoregressivo.")
    if all_xgb_native_importance:
        print(f"File scritto: {xgb_native_path}")
        print(f"File scritto: {xgb_native_summary_path}")
    if not extended_single_summary_out.empty:
        print(f"File scritto: {extended_single_summary_path}")
    if not extended_single_delta_summary.empty:
        print(f"File scritto: {extended_single_delta_summary_path}")
    if not args.no_multioutput_xgb and 'extended_multi_summary_out' in locals() and not extended_multi_summary_out.empty:
        print(f"File scritto: {extended_multi_summary_path}")
    if runtime_rows:
        print(f"File scritto: {runtime_profile_path}")
    if not args.no_multioutput_xgb and 'extended_multi_delta_summary' in locals() and not extended_multi_delta_summary.empty:
        print(f"File scritto: {extended_multi_delta_summary_path}")
    return 0


def main() -> int:
    args = parse_args()
    return run_analysis(args)


if __name__ == "__main__":
    sys.exit(main())
