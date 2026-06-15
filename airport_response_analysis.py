#!/usr/bin/env python3
"""Explanatory response analysis for BLQ activity, wind regime, and pollutants."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from analysis_runtime import resolve_workers

from upwind_downwind_analysis import (
    AIRPORT_REFERENCE,
    DATETIME_COLUMN,
    DEFAULT_INPUT,
    STATIONS,
    SPATIAL_GRADIENTS,
    POLLUTANTS,
    add_control_features,
    add_multistation_regimes,
    add_spatial_gradients,
    add_station_wind_features,
    classify_wind_regime,
    read_dataset,
    target_station_id,
)


DEFAULT_OUTPUT_DIR = Path("Analysis/airport_response")
PLOT_FORMAT = "svg"
BLQ_SERVICE_FEATURES = [
    "blq_service_scheduled_passenger_flights",
    "blq_service_charter_passenger_flights",
    "blq_service_cargo_flights",
    "blq_service_mail_flights",
    "blq_service_combined_flights",
]
CONTEXT_ORDER = {
    "all": ["all"],
    "urban_traffic": ["low_urban_traffic", "high_urban_traffic"],
    "season": ["cold_season", "warm_season"],
    "daypart": ["daytime", "nighttime"],
}
WIND_REGIMES = ["downwind", "upwind", "crosswind"]
CONTEXT_SPECS = [
    ("all", ["all"]),
    ("urban_traffic", CONTEXT_ORDER["urban_traffic"]),
    ("season", CONTEXT_ORDER["season"]),
    ("daypart", CONTEXT_ORDER["daypart"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analisi esplicativa aggiuntiva su risposta dei target all'attivita' BLQ, "
            "regime di vento, eventi ad alta attivita' e gradienti spaziali."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", nargs="+", default=POLLUTANTS)
    parser.add_argument("--blq-activity", default="blq_flights")
    parser.add_argument("--alignment-threshold", type=float, default=0.5)
    parser.add_argument("--min-wind-component", type=float, default=0.5)
    parser.add_argument("--wind-source", choices=["aero", "centro"], default="aero")
    parser.add_argument("--blq-bins", type=int, default=5)
    parser.add_argument("--high-blq-quantile", type=float, default=0.9)
    parser.add_argument("--high-urban-quantile", type=float, default=0.9)
    parser.add_argument("--event-window-hours", type=int, default=24)
    parser.add_argument("--max-events-per-target", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--partial-grid-size", type=int, default=25)
    parser.add_argument("--partial-sample", type=int, default=2000)
    parser.add_argument("--plot-top-events", type=int, default=3)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Parallelismo locale. 0=auto: usa SLURM_CPUS_PER_TASK se presente, "
            "altrimenti os.cpu_count()."
        ),
    )
    return parser.parse_args()


def enrich_context(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["blq_activity"] = out[args.blq_activity]
    urban_q50 = out["urban_traffic_total"].quantile(0.5)
    out["urban_traffic_level"] = np.where(
        out["urban_traffic_total"] <= urban_q50,
        "low_urban_traffic",
        "high_urban_traffic",
    )
    month = out[DATETIME_COLUMN].dt.month
    out["season_group"] = np.where(month.isin([4, 5, 6, 7, 8, 9]), "warm_season", "cold_season")
    hour = out[DATETIME_COLUMN].dt.hour
    out["daypart_group"] = np.where(((hour >= 7) & (hour < 21)), "daytime", "nighttime")
    return out


def _safe_qcut(series: pd.Series, q: int) -> pd.Series:
    ranked = series.rank(method="first")
    try:
        return pd.qcut(ranked, q=q, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(np.zeros(len(series), dtype=int), index=series.index)


def _plots_dir(output_dir: Path) -> Path:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


_GLOBAL_DF: pd.DataFrame | None = None
_GLOBAL_ARGS: argparse.Namespace | None = None


def _init_worker(df: pd.DataFrame, args: argparse.Namespace) -> None:
    global _GLOBAL_DF, _GLOBAL_ARGS
    _GLOBAL_DF = df
    _GLOBAL_ARGS = args


def target_station_feature_columns(target: str, wind_source: str) -> dict[str, str]:
    station_id = target_station_id(target)
    if station_id is None:
        station_id = "porta_san_felice"
    return {
        "station_id": station_id,
        "alignment": f"airport_to_{station_id}_wind_alignment_{wind_source}",
        "component": f"airport_to_{station_id}_wind_component_{wind_source}",
        "crosswind": f"airport_to_{station_id}_crosswind_component_{wind_source}",
        "regime": f"{station_id}_wind_regime",
        "downwind_flag": f"{station_id}_downwind_flag",
        "upwind_flag": f"{station_id}_upwind_flag",
    }


def _subset_pairs(regime_df: pd.DataFrame, context_type: str, values: list[str]) -> list[tuple[str, pd.DataFrame]]:
    if context_type == "all":
        return [("all", regime_df)]
    if context_type == "urban_traffic":
        column = "urban_traffic_level"
    elif context_type == "season":
        column = "season_group"
    else:
        column = "daypart_group"
    return [(value, regime_df.loc[regime_df[column] == value].copy()) for value in values]


def _empirical_response_curves_for_target(df: pd.DataFrame, target: str, bins: int) -> pd.DataFrame:
    rows: list[dict] = []
    if target not in df.columns:
        return pd.DataFrame()
    station_cols = target_station_feature_columns(target, _GLOBAL_ARGS.wind_source if _GLOBAL_ARGS else "aero")
    regime_col = station_cols["regime"]
    if regime_col not in df.columns:
        return pd.DataFrame()
    clean = df.dropna(subset=[target, "blq_activity"]).copy()
    if clean.empty:
        return pd.DataFrame()
    for regime in WIND_REGIMES:
        regime_df = clean.loc[clean[regime_col] == regime].copy()
        if regime_df.empty:
            continue
        for context_type, values in CONTEXT_SPECS:
            for context_value, subset in _subset_pairs(regime_df, context_type, values):
                if subset.empty or subset["blq_activity"].nunique() < 2:
                    continue
                subset["blq_bin"] = _safe_qcut(subset["blq_activity"], bins)
                grouped = subset.groupby("blq_bin", dropna=True)
                for bin_id, group in grouped:
                    rows.append(
                        {
                            "target": target,
                            "wind_regime": regime,
                            "context_type": context_type,
                            "context_value": context_value,
                            "blq_bin": int(bin_id) + 1,
                            "rows": len(group),
                            "blq_mean": group["blq_activity"].mean(),
                            "blq_p25": group["blq_activity"].quantile(0.25),
                            "blq_p75": group["blq_activity"].quantile(0.75),
                            "target_mean": group[target].mean(),
                            "target_median": group[target].median(),
                            "target_p25": group[target].quantile(0.25),
                            "target_p75": group[target].quantile(0.75),
                        }
                    )
    return pd.DataFrame(rows)


def empirical_response_curves(df: pd.DataFrame, targets: list[str], bins: int) -> pd.DataFrame:
    frames = [_empirical_response_curves_for_target(df, target, bins) for target in targets]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _run_empirical_target(target: str) -> pd.DataFrame:
    return _empirical_response_curves_for_target(_GLOBAL_DF, target, _GLOBAL_ARGS.blq_bins)


def save_empirical_response_plots(curves: pd.DataFrame, output_dir: Path) -> None:
    if curves.empty:
        return
    plots_dir = _plots_dir(output_dir)
    colors = {"downwind": "#d62728", "upwind": "#1f77b4", "crosswind": "#7f7f7f"}
    contexts_to_plot = [context_type for context_type, _ in CONTEXT_SPECS]

    for target in sorted(curves["target"].unique()):
        target_df = curves.loc[curves["target"] == target].copy()
        for context_type in contexts_to_plot:
            context_df = target_df.loc[target_df["context_type"] == context_type].copy()
            if context_df.empty:
                continue
            context_values = CONTEXT_ORDER.get(context_type, ["all"])
            ncols = 1 if context_type == "all" else len(context_values)
            fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.4), constrained_layout=False)
            if ncols == 1:
                axes = [axes]
            for ax, context_value in zip(axes, context_values):
                subset = context_df.loc[context_df["context_value"] == context_value].copy()
                if subset.empty:
                    ax.set_axis_off()
                    continue
                for regime in WIND_REGIMES:
                    regime_df = subset.loc[subset["wind_regime"] == regime].sort_values("blq_mean")
                    if regime_df.empty:
                        continue
                    ax.plot(
                        regime_df["blq_mean"],
                        regime_df["target_mean"],
                        marker="o",
                        linewidth=2,
                        color=colors[regime],
                        label=regime,
                    )
                ax.set_title(context_value.replace("_", " "))
                ax.set_xlabel("Attivita' BLQ media nel bin")
                ax.set_ylabel("Concentrazione media target")
                ax.grid(True, alpha=0.3)
            handles, labels = [], []
            for ax in axes:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    break
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.98),
                    ncol=3,
                    frameon=True,
                )
            fig.suptitle(f"{target} - risposta empirica a BLQ ({context_type})", y=1.06)
            fig.subplots_adjust(top=0.82)
            fig.savefig(
                plots_dir / f"{target}_empirical_response_{context_type}.{PLOT_FORMAT}",
                bbox_inches="tight",
                format=PLOT_FORMAT,
            )
            plt.close(fig)


def explanatory_feature_set(df: pd.DataFrame, target: str, wind_source: str) -> list[str]:
    station_cols = target_station_feature_columns(target, wind_source)
    candidates = [
        "blq_activity",
        *BLQ_SERVICE_FEATURES,
        "urban_traffic_total",
        "airport_spire_traffic_total",
        station_cols["alignment"],
        station_cols["component"],
        station_cols["crosswind"],
        station_cols["downwind_flag"],
        station_cols["upwind_flag"],
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
        "night_flag",
        "low_ventilation_flag",
    ]
    return [col for col in candidates if col in df.columns]


def temporal_train_test(data: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = int(len(data) * (1 - test_size))
    split = max(1, min(split, len(data) - 1))
    return data.iloc[:split].copy(), data.iloc[split:].copy()


def fit_explanatory_model_profiles(
    df: pd.DataFrame,
    targets: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict] = []
    profile_rows: list[dict] = []
    for target in targets:
        if target not in df.columns:
            continue
        station_cols = target_station_feature_columns(target, args.wind_source)
        selected_features = [
            "blq_activity",
            "blq_service_cargo_flights",
            "blq_service_scheduled_passenger_flights",
        ] + [
            station_cols["alignment"],
            station_cols["component"],
            station_cols["downwind_flag"],
            station_cols["upwind_flag"],
        ]
        selected_features = [f for f in selected_features if f in df.columns]
        features = explanatory_feature_set(df, target, args.wind_source)
        clean = df.dropna(subset=[target]).copy()
        if clean.empty or len(features) < 3:
            continue
        train, test = temporal_train_test(clean, args.test_size)
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
            n_jobs=1,
        )
        model.fit(x_train, train[target])
        preds = model.predict(x_test)
        metrics_rows.append(
            {
                "target": target,
                "rows_train": len(train),
                "rows_test": len(test),
                "r2_test": r2_score(test[target], preds),
                "mae_test": mean_absolute_error(test[target], preds),
                "rmse_test": float(np.sqrt(mean_squared_error(test[target], preds))),
                "n_features": len(features),
            }
        )

        reference = clean[features].sample(n=min(args.partial_sample, len(clean)), random_state=42).copy()
        reference_imputed = pd.DataFrame(imputer.transform(reference), columns=features)
        for feature in selected_features:
            if feature not in reference_imputed.columns:
                continue
            series = reference_imputed[feature]
            if feature.endswith("_flag"):
                grid = np.array([0.0, 1.0])
            else:
                low = float(series.quantile(0.05))
                high = float(series.quantile(0.95))
                if not np.isfinite(low) or not np.isfinite(high) or low == high:
                    continue
                grid = np.linspace(low, high, args.partial_grid_size)
            col_idx = reference_imputed.columns.get_loc(feature)
            base_matrix = reference_imputed.to_numpy(copy=True)
            for value in grid:
                modified = base_matrix.copy()
                modified[:, col_idx] = value
                pred = model.predict(modified)
                profile_rows.append(
                    {
                        "target": target,
                        "feature": feature,
                        "grid_value": float(value),
                        "mean_prediction": float(np.mean(pred)),
                        "p10_prediction": float(np.quantile(pred, 0.10)),
                        "p90_prediction": float(np.quantile(pred, 0.90)),
                    }
                )
    return pd.DataFrame(metrics_rows), pd.DataFrame(profile_rows)


def _fit_explanatory_model_profiles_for_target(
    df: pd.DataFrame,
    target: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, profiles = fit_explanatory_model_profiles(df, [target], args)
    return metrics, profiles


def _run_model_target(target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _fit_explanatory_model_profiles_for_target(_GLOBAL_DF, target, _GLOBAL_ARGS)


def save_partial_dependence_plots(profiles: pd.DataFrame, output_dir: Path, wind_source: str) -> None:
    if profiles.empty:
        return
    plots_dir = _plots_dir(output_dir)
    feature_order_template = [
        "blq_activity",
        "blq_service_cargo_flights",
        "blq_service_scheduled_passenger_flights",
    ]
    for target in sorted(profiles["target"].unique()):
        target_df = profiles.loc[profiles["target"] == target].copy()
        station_cols = target_station_feature_columns(target, wind_source)
        feature_order = feature_order_template + [
            station_cols["alignment"],
            station_cols["component"],
            station_cols["downwind_flag"],
            station_cols["upwind_flag"],
        ]
        feature_order = [f for f in feature_order if f in target_df["feature"].unique()]
        if not feature_order:
            continue
        ncols = 2
        nrows = int(np.ceil(len(feature_order) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, feature in zip(axes, feature_order):
            feat_df = target_df.loc[target_df["feature"] == feature].sort_values("grid_value")
            ax.plot(feat_df["grid_value"], feat_df["mean_prediction"], color="#1f77b4", linewidth=2)
            ax.fill_between(
                feat_df["grid_value"],
                feat_df["p10_prediction"],
                feat_df["p90_prediction"],
                color="#1f77b4",
                alpha=0.15,
            )
            ax.set_title(feature)
            ax.set_xlabel("Valore della feature")
            ax.set_ylabel("Predizione media")
            ax.grid(True, alpha=0.3)
        for ax in axes[len(feature_order):]:
            ax.set_axis_off()
        fig.suptitle(f"{target} - profili di dipendenza parziale", y=1.02)
        fig.savefig(plots_dir / f"{target}_partial_dependence.{PLOT_FORMAT}", bbox_inches="tight", format=PLOT_FORMAT)
        plt.close(fig)


def event_windows(
    df: pd.DataFrame,
    targets: list[str],
    args: argparse.Namespace,
    reference_targets: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows: list[dict] = []
    window_rows: list[pd.DataFrame] = []
    available_targets = reference_targets if reference_targets is not None else targets
    urban_limit = df["urban_traffic_total"].quantile(args.high_urban_quantile)
    gap_limit = pd.Timedelta(hours=6)
    for target in targets:
        if target not in df.columns:
            continue
        station_cols = target_station_feature_columns(target, args.wind_source)
        regime_col = station_cols["regime"]
        downwind_col = station_cols["downwind_flag"]
        component_col = station_cols["component"]
        alignment_col = station_cols["alignment"]
        if downwind_col not in df.columns or regime_col not in df.columns:
            continue
        high_blq = df["blq_activity"].quantile(args.high_blq_quantile)
        candidates = df.loc[
            (df[downwind_col] == 1)
            & (df["blq_activity"] >= high_blq)
            & (df["urban_traffic_total"] <= urban_limit)
            & df[target].notna()
        ].copy()
        if candidates.empty:
            continue
        candidates = candidates.sort_values(DATETIME_COLUMN)
        event_break = candidates[DATETIME_COLUMN].diff().gt(gap_limit).fillna(True)
        candidates["event_id"] = event_break.cumsum()
        event_summary = (
            candidates.groupby("event_id", as_index=False)
            .agg(
                start=(DATETIME_COLUMN, "min"),
                end=(DATETIME_COLUMN, "max"),
                peak_blq_activity=("blq_activity", "max"),
                mean_blq_activity=("blq_activity", "mean"),
                mean_target=(target, "mean"),
                peak_target=(target, "max"),
                peak_alignment=(alignment_col, "max"),
                peak_component=(component_col, "max"),
                rows=("event_id", "size"),
            )
            .sort_values("peak_blq_activity", ascending=False)
            .head(args.max_events_per_target)
        )
        for rank, row in enumerate(event_summary.itertuples(index=False), start=1):
            event_label = f"{target}_event_{rank:02d}"
            event_rows.append(
                {
                    "target": target,
                    "event_label": event_label,
                    "start": row.start,
                    "end": row.end,
                    "peak_blq_activity": row.peak_blq_activity,
                    "mean_blq_activity": row.mean_blq_activity,
                    "mean_target": row.mean_target,
                    "peak_target": row.peak_target,
                    "peak_alignment": row.peak_alignment,
                    "peak_component": row.peak_component,
                    "rows": row.rows,
                }
            )
            center = row.start + (row.end - row.start) / 2
            window = df.loc[
                (df[DATETIME_COLUMN] >= center - pd.Timedelta(hours=args.event_window_hours))
                & (df[DATETIME_COLUMN] <= center + pd.Timedelta(hours=args.event_window_hours))
            ].copy()
            pollutant_prefix = target.split("_")[0]
            peer_targets = [
                column
                for column in available_targets
                if column != target and column.startswith(f"{pollutant_prefix}_") and column in window.columns
            ]
            keep_cols = [
                DATETIME_COLUMN,
                target,
                *peer_targets,
                "blq_activity",
                "urban_traffic_total",
                alignment_col,
                component_col,
                regime_col,
                downwind_col,
            ]
            window["target"] = target
            window["event_label"] = event_label
            window_rows.append(window[["target", "event_label", *keep_cols]].copy())
    summary = pd.DataFrame(event_rows)
    windows = pd.concat(window_rows, ignore_index=True) if window_rows else pd.DataFrame()
    return summary, windows


def _event_windows_for_target(
    df: pd.DataFrame,
    target: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return event_windows(df, [target], args, reference_targets=args.targets)


def _run_event_target(target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _event_windows_for_target(_GLOBAL_DF, target, _GLOBAL_ARGS)


def save_event_plots(
    events: pd.DataFrame,
    windows: pd.DataFrame,
    output_dir: Path,
    plot_top_events: int,
    wind_source: str,
) -> None:
    if events.empty or windows.empty:
        return
    plots_dir = _plots_dir(output_dir)
    for target in sorted(events["target"].unique()):
        event_subset = events.loc[events["target"] == target].head(plot_top_events)
        for row in event_subset.itertuples(index=False):
            event_window = windows.loc[windows["event_label"] == row.event_label].copy()
            if event_window.empty:
                continue
            event_window = event_window.sort_values(DATETIME_COLUMN)
            target_col = target
            peer_cols = [
                c
                for c in event_window.columns
                if c.startswith(target.split("_")[0] + "_") and c != target_col and event_window[c].notna().any()
            ]
            station_cols = target_station_feature_columns(target, wind_source)
            alignment_col = station_cols["alignment"]
            component_col = station_cols["component"]

            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
            axes[0].plot(event_window[DATETIME_COLUMN], event_window[target_col], color="#d62728", linewidth=2, label=target_col)
            for peer in peer_cols[:2]:
                axes[0].plot(event_window[DATETIME_COLUMN], event_window[peer], linewidth=1.3, alpha=0.8, label=peer)
            axes[0].set_ylabel("Concentrazione")
            axes[0].set_title(f"{row.event_label} - finestra evento")
            axes[0].legend(fontsize=8)
            axes[0].grid(True, alpha=0.3)

            ax_blq = axes[1]
            ax_traffic = ax_blq.twinx()
            blq_line = ax_blq.plot(
                event_window[DATETIME_COLUMN],
                event_window["blq_activity"],
                color="#1f77b4",
                linewidth=2,
                label="BLQ",
            )[0]
            traffic_line = ax_traffic.plot(
                event_window[DATETIME_COLUMN],
                event_window["urban_traffic_total"],
                color="#2ca02c",
                linewidth=1.5,
                label="Traffico urbano",
            )[0]
            ax_blq.set_ylabel("BLQ")
            ax_traffic.set_ylabel("Traffico urbano")
            ax_blq.legend([blq_line, traffic_line], ["BLQ", "Traffico urbano"], fontsize=8, loc="upper left")
            axes[1].grid(True, alpha=0.3)

            if component_col in event_window.columns and event_window[component_col].notna().any():
                axes[2].plot(event_window[DATETIME_COLUMN], event_window[component_col], color="#9467bd", linewidth=2, label="Wind component")
            if alignment_col in event_window.columns and event_window[alignment_col].notna().any():
                axes[2].plot(event_window[DATETIME_COLUMN], event_window[alignment_col], color="#ff7f0e", linewidth=1.5, label="Wind alignment")
            axes[2].axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.8)
            axes[2].set_ylabel("Vento")
            axes[2].legend(fontsize=8)
            axes[2].grid(True, alpha=0.3)

            fig.savefig(plots_dir / f"{row.event_label}.{PLOT_FORMAT}", bbox_inches="tight", format=PLOT_FORMAT)
            plt.close(fig)


def exceedance_probabilities(df: pd.DataFrame, targets: list[str], bins: int) -> pd.DataFrame:
    rows: list[dict] = []
    for target in targets:
        if target not in df.columns:
            continue
        station_cols = target_station_feature_columns(target, _GLOBAL_ARGS.wind_source if _GLOBAL_ARGS else "aero")
        regime_col = station_cols["regime"]
        if regime_col not in df.columns:
            continue
        clean = df.dropna(subset=[target, "blq_activity"]).copy()
        if clean.empty or clean["blq_activity"].nunique() < 2:
            continue
        thresholds = {
            "p75": clean[target].quantile(0.75),
            "p90": clean[target].quantile(0.90),
        }
        clean["blq_bin"] = _safe_qcut(clean["blq_activity"], bins)
        for threshold_name, threshold_value in thresholds.items():
            for regime in WIND_REGIMES:
                subset = clean.loc[clean[regime_col] == regime]
                if subset.empty:
                    continue
                grouped = subset.groupby("blq_bin", dropna=True)
                for bin_id, group in grouped:
                    rows.append(
                        {
                            "target": target,
                            "threshold_name": threshold_name,
                            "threshold_value": threshold_value,
                            "wind_regime": regime,
                            "blq_bin": int(bin_id) + 1,
                            "rows": len(group),
                            "blq_mean": group["blq_activity"].mean(),
                            "prob_exceedance": float((group[target] >= threshold_value).mean()),
                        }
                    )
    return pd.DataFrame(rows)


def _exceedance_probabilities_for_target(df: pd.DataFrame, target: str, bins: int) -> pd.DataFrame:
    return exceedance_probabilities(df, [target], bins)


def _run_exceedance_target(target: str) -> pd.DataFrame:
    return _exceedance_probabilities_for_target(_GLOBAL_DF, target, _GLOBAL_ARGS.blq_bins)


def save_exceedance_plots(exceedance: pd.DataFrame, output_dir: Path) -> None:
    if exceedance.empty:
        return
    plots_dir = _plots_dir(output_dir)
    colors = {"downwind": "#d62728", "upwind": "#1f77b4", "crosswind": "#7f7f7f"}
    for target in sorted(exceedance["target"].unique()):
        target_df = exceedance.loc[exceedance["target"] == target]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, threshold_name in zip(axes, ["p75", "p90"]):
            subset = target_df.loc[target_df["threshold_name"] == threshold_name]
            for regime in WIND_REGIMES:
                regime_df = subset.loc[subset["wind_regime"] == regime].sort_values("blq_mean")
                if regime_df.empty:
                    continue
                ax.plot(
                    regime_df["blq_mean"],
                    regime_df["prob_exceedance"],
                    marker="o",
                    linewidth=2,
                    color=colors[regime],
                    label=regime,
                )
            ax.set_title(threshold_name)
            ax.set_xlabel("Attivita' BLQ media nel bin")
            ax.set_ylabel("Probabilita' di superamento")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=3)
        fig.suptitle(f"{target} - probabilita' di superamento soglia", y=1.02)
        fig.savefig(plots_dir / f"{target}_exceedance_probability.{PLOT_FORMAT}", bbox_inches="tight", format=PLOT_FORMAT)
        plt.close(fig)


def descriptive_spatial_gradients(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    rows: list[dict] = []
    for gradient, (left_target, right_target) in SPATIAL_GRADIENTS.items():
        if gradient not in df.columns:
            continue
        left_station = target_station_id(left_target)
        right_station = target_station_id(right_target)
        if left_station is None or right_station is None:
            continue
        left_regime_col = f"{left_station}_wind_regime"
        right_regime_col = f"{right_station}_wind_regime"
        if left_regime_col not in df.columns or right_regime_col not in df.columns:
            continue
        clean = df.dropna(subset=[gradient, "blq_activity"]).copy()
        if clean.empty or clean["blq_activity"].nunique() < 2:
            continue
        clean["blq_bin"] = _safe_qcut(clean["blq_activity"], bins)
        for regime in WIND_REGIMES:
            subset = clean.loc[(clean[left_regime_col] == regime) & (clean[right_regime_col] == regime)]
            if subset.empty:
                continue
            grouped = subset.groupby("blq_bin", dropna=True)
            for bin_id, group in grouped:
                rows.append(
                    {
                        "gradient": gradient,
                        "wind_regime": regime,
                        "blq_bin": int(bin_id) + 1,
                        "rows": len(group),
                        "blq_mean": group["blq_activity"].mean(),
                        "gradient_mean": group[gradient].mean(),
                        "gradient_median": group[gradient].median(),
                        "gradient_p25": group[gradient].quantile(0.25),
                        "gradient_p75": group[gradient].quantile(0.75),
                    }
                )
    return pd.DataFrame(rows)


def save_gradient_plots(gradients: pd.DataFrame, output_dir: Path) -> None:
    if gradients.empty:
        return
    plots_dir = _plots_dir(output_dir)
    colors = {"downwind": "#d62728", "upwind": "#1f77b4", "crosswind": "#7f7f7f"}
    for gradient in sorted(gradients["gradient"].unique()):
        gradient_df = gradients.loc[gradients["gradient"] == gradient]
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for regime in WIND_REGIMES:
            regime_df = gradient_df.loc[gradient_df["wind_regime"] == regime].sort_values("blq_mean")
            if regime_df.empty:
                continue
            ax.plot(
                regime_df["blq_mean"],
                regime_df["gradient_mean"],
                marker="o",
                linewidth=2,
                color=colors[regime],
                label=regime,
            )
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.8)
        ax.set_title(f"{gradient} - gradienti per classi di BLQ")
        ax.set_xlabel("Attivita' BLQ media nel bin")
        ax.set_ylabel("Gradiente medio")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.savefig(plots_dir / f"{gradient}_descriptive_gradient.{PLOT_FORMAT}", bbox_inches="tight", format=PLOT_FORMAT)
        plt.close(fig)


def _run_parallel_targets(
    targets: list[str],
    worker_fn,
    args: argparse.Namespace,
) -> list:
    valid_targets = [target for target in targets if target in _GLOBAL_DF.columns]
    if not valid_targets:
        return []
    workers = resolve_workers(args.workers, len(valid_targets))
    print(f"Worker {worker_fn.__name__}: {workers}")
    if workers == 1:
        return [worker_fn(target) for target in valid_targets]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(_GLOBAL_DF, _GLOBAL_ARGS),
    ) as executor:
        futures = {executor.submit(worker_fn, target): target for target in valid_targets}
        return [future.result() for future in as_completed(futures)]


def main() -> int:
    args = parse_args()
    df = read_dataset(args.input)
    df = add_station_wind_features(df)
    df = classify_wind_regime(df, args)
    df = add_multistation_regimes(df, args)
    df = add_control_features(df)
    df = add_spatial_gradients(df)
    df = enrich_context(df, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _init_worker(df, args)

    curve_frames = _run_parallel_targets(args.targets, _run_empirical_target, args)
    curves = pd.concat([frame for frame in curve_frames if not frame.empty], ignore_index=True) if curve_frames else pd.DataFrame()

    model_results = _run_parallel_targets(args.targets, _run_model_target, args)
    metrics_frames = [metrics for metrics, _ in model_results if not metrics.empty]
    profile_frames = [profiles for _, profiles in model_results if not profiles.empty]
    model_metrics = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    profiles = pd.concat(profile_frames, ignore_index=True) if profile_frames else pd.DataFrame()

    event_results = _run_parallel_targets(args.targets, _run_event_target, args)
    event_frames = [events_part for events_part, _ in event_results if not events_part.empty]
    window_frames = [windows_part for _, windows_part in event_results if not windows_part.empty]
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    windows = pd.concat(window_frames, ignore_index=True) if window_frames else pd.DataFrame()

    exceedance_frames = _run_parallel_targets(args.targets, _run_exceedance_target, args)
    exceedance = (
        pd.concat([frame for frame in exceedance_frames if not frame.empty], ignore_index=True)
        if exceedance_frames
        else pd.DataFrame()
    )
    gradients = descriptive_spatial_gradients(df, args.blq_bins)

    curves_path = args.output_dir / "blq_empirical_response_curves.csv"
    metrics_path = args.output_dir / "blq_partial_dependence_model_metrics.csv"
    profiles_path = args.output_dir / "blq_partial_dependence_profiles.csv"
    events_path = args.output_dir / "blq_event_windows_summary.csv"
    windows_path = args.output_dir / "blq_event_windows_long.csv"
    exceedance_path = args.output_dir / "blq_exceedance_probabilities.csv"
    gradients_path = args.output_dir / "blq_spatial_gradient_response.csv"

    curves.to_csv(curves_path, index=False)
    model_metrics.to_csv(metrics_path, index=False)
    profiles.to_csv(profiles_path, index=False)
    events.to_csv(events_path, index=False)
    windows.to_csv(windows_path, index=False)
    exceedance.to_csv(exceedance_path, index=False)
    gradients.to_csv(gradients_path, index=False)

    save_empirical_response_plots(curves, args.output_dir)
    save_partial_dependence_plots(profiles, args.output_dir, args.wind_source)
    save_event_plots(events, windows, args.output_dir, args.plot_top_events, args.wind_source)
    save_exceedance_plots(exceedance, args.output_dir)
    save_gradient_plots(gradients, args.output_dir)

    print(f"File scritto: {curves_path}")
    print(f"File scritto: {metrics_path}")
    print(f"File scritto: {profiles_path}")
    print(f"File scritto: {events_path}")
    print(f"File scritto: {windows_path}")
    print(f"File scritto: {exceedance_path}")
    print(f"File scritto: {gradients_path}")
    print(f"Cartella grafici: {args.output_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
