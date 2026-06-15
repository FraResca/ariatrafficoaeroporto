#!/usr/bin/env python3
"""Build explicit cross-pollutant comparisons from existing analysis outputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_runtime import resolve_workers


DEFAULT_DATASET = Path("Datasets_Raw/hourly_merged_2023_2025.csv")
DEFAULT_EXPLAIN_DIR = Path("Analysis/slurm_full_explain")
DEFAULT_UPWIND_DIR = Path("Analysis/slurm_full_upwind")
DEFAULT_OUTPUT_DIR = Path("Analysis/cross_pollutant")
PLOT_FORMAT = "svg"
TARGET_ORDER = [
    "NO2_porta_san_felice",
    "CO_porta_san_felice",
    "C6H6_porta_san_felice",
    "NO2_giardini_margherita",
    "NO2_via_chiarini",
    "O3_giardini_margherita",
    "O3_via_chiarini",
]
GROUP_ORDER = [
    "meteo",
    "other_pollutants",
    "rolling_features",
    "urban_traffic",
    "wind_transport",
    "airport",
    "airport_service_type",
    "station_wind_bools",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Costruisce un confronto esplicito tra inquinanti a partire dagli output "
            "gia' prodotti dalle analisi explain e upwind/downwind."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Dataset orario unificato.")
    parser.add_argument("--explain-dir", type=Path, default=DEFAULT_EXPLAIN_DIR, help="Cartella output explain.")
    parser.add_argument("--upwind-dir", type=Path, default=DEFAULT_UPWIND_DIR, help="Cartella output upwind/downwind.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Cartella output confronto.")
    parser.add_argument("--workers", type=int, default=0, help="Parallelismo locale. 0=auto.")
    parser.add_argument("--no-progress", action="store_true", help="Riduce i messaggi intermedi.")
    return parser.parse_args()


def split_target(target: str) -> tuple[str, str]:
    pollutant, station = target.split("_", 1)
    return pollutant, station


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    return pd.read_csv(path)


def load_inputs(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    return {
        "dataset": read_csv(args.dataset),
        "summary": read_csv(args.explain_dir / "advanced_temporal_cv_summary.csv"),
        "extended_ablation": read_csv(args.explain_dir / "advanced_extended_ablation_delta_summary.csv"),
        "targeted_ablation": read_csv(args.explain_dir / "advanced_ablation_summary.csv"),
        "group_shap": read_csv(args.explain_dir / "advanced_group_shap.csv"),
        "upwind_summary": read_csv(args.upwind_dir / "upwind_downwind_summary.csv"),
        "matched": read_csv(args.upwind_dir / "upwind_downwind_matched_summary.csv"),
        "bootstrap": read_csv(args.upwind_dir / "upwind_downwind_bootstrap_effects.csv"),
        "threshold": read_csv(args.upwind_dir / "upwind_downwind_threshold_sensitivity.csv"),
    }


def target_std_from_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in TARGET_ORDER:
        if target not in df.columns:
            continue
        pollutant, station = split_target(target)
        values = pd.to_numeric(df[target], errors="coerce").dropna()
        rows.append(
            {
                "target": target,
                "pollutant": pollutant,
                "station": station,
                "target_mean": float(values.mean()) if not values.empty else np.nan,
                "target_std": float(values.std(ddof=0)) if not values.empty else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Nessun target presente nel dataset raw.")
    return out


def _best_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    return df.sort_values(["mean_R2", "mean_RMSE"], ascending=[False, True]).iloc[0]


def build_predictability_outputs(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for target in TARGET_ORDER:
        target_df = summary.loc[summary["target"] == target]
        if target_df.empty:
            continue
        pollutant, station = split_target(target)
        for horizon in sorted(target_df["horizon_h"].unique()):
            subset = target_df.loc[target_df["horizon_h"] == horizon]
            no_auto = subset.loc[subset["feature_set"].str.startswith("no_target_")]
            with_auto = subset.loc[subset["feature_set"].str.startswith("with_target_")]
            overall = subset
            best_no = _best_row(no_auto)
            best_with = _best_row(with_auto)
            best_overall = _best_row(overall)
            rows.append(
                {
                    "target": target,
                    "pollutant": pollutant,
                    "station": station,
                    "horizon_h": int(horizon),
                    "best_overall_model": None if best_overall is None else best_overall["model"],
                    "best_overall_feature_set": None if best_overall is None else best_overall["feature_set"],
                    "best_overall_R2": np.nan if best_overall is None else float(best_overall["mean_R2"]),
                    "best_no_auto_model": None if best_no is None else best_no["model"],
                    "best_no_auto_feature_set": None if best_no is None else best_no["feature_set"],
                    "best_no_auto_R2": np.nan if best_no is None else float(best_no["mean_R2"]),
                    "best_with_auto_model": None if best_with is None else best_with["model"],
                    "best_with_auto_feature_set": None if best_with is None else best_with["feature_set"],
                    "best_with_auto_R2": np.nan if best_with is None else float(best_with["mean_R2"]),
                    "best_no_auto_MAE": np.nan if best_no is None else float(best_no["mean_MAE"]),
                    "best_with_auto_MAE": np.nan if best_with is None else float(best_with["mean_MAE"]),
                    "autoregressive_gain_R2": np.nan if best_no is None or best_with is None else float(best_with["mean_R2"] - best_no["mean_R2"]),
                    "autoregressive_gain_MAE_reduction": np.nan if best_no is None or best_with is None else float(best_no["mean_MAE"] - best_with["mean_MAE"]),
                }
            )
    predictability = pd.DataFrame(rows).sort_values(["target", "horizon_h"])
    no_auto_pivot = (
        predictability.pivot_table(
            index=["target", "pollutant", "station"],
            columns="horizon_h",
            values="best_no_auto_R2",
            aggfunc="first",
        )
        .rename(columns=lambda h: f"best_no_auto_R2_{int(h)}h")
        .reset_index()
    )
    with_auto_pivot = (
        predictability.pivot_table(
            index=["target", "pollutant", "station"],
            columns="horizon_h",
            values="best_with_auto_R2",
            aggfunc="first",
        )
        .rename(columns=lambda h: f"best_with_auto_R2_{int(h)}h")
        .reset_index()
    )
    target_overview = (
        predictability.groupby(["target", "pollutant", "station"], as_index=False)
        .agg(
            mean_autoregressive_gain_R2=("autoregressive_gain_R2", "mean"),
            max_autoregressive_gain_R2=("autoregressive_gain_R2", "max"),
            mean_best_overall_R2=("best_overall_R2", "mean"),
        )
        .merge(no_auto_pivot, on=["target", "pollutant", "station"], how="left")
        .merge(with_auto_pivot, on=["target", "pollutant", "station"], how="left")
    )
    pollutant_overview = (
        target_overview.groupby("pollutant", as_index=False)
        .agg(
            n_targets=("target", "count"),
            mean_best_no_auto_R2_1h=("best_no_auto_R2_1h", "mean"),
            mean_best_no_auto_R2_24h=("best_no_auto_R2_24h", "mean"),
            mean_best_with_auto_R2_1h=("best_with_auto_R2_1h", "mean"),
            mean_best_with_auto_R2_24h=("best_with_auto_R2_24h", "mean"),
            mean_autoregressive_gain_R2=("mean_autoregressive_gain_R2", "mean"),
            mean_best_overall_R2=("mean_best_overall_R2", "mean"),
        )
    )
    return predictability, target_overview, pollutant_overview


def build_ablation_outputs(
    extended: pd.DataFrame,
    targeted: pd.DataFrame,
    shap: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ext = extended.loc[
        (extended["analysis_scope"] == "single_target")
        & (extended["analysis_type"] == "drop_one_group")
        & (extended["model"] == "xgboost")
        & (extended["baseline_feature_set"] == "no_target_autoregressive")
    ].copy()
    ext["group"] = ext["removed_groups"]
    ext = ext.loc[ext["group"].isin(GROUP_ORDER)]
    ext_group = (
        ext.groupby(["target", "group"], as_index=False)
        .agg(mean_delta_R2=("delta_R2_mean", "mean"))
    )
    ext_group[["pollutant", "station"]] = ext_group["target"].apply(lambda t: pd.Series(split_target(t)))
    ext_pivot = (
        ext_group.pivot(index="target", columns="group", values="mean_delta_R2")
        .reindex(index=[t for t in TARGET_ORDER if t in ext_group["target"].unique()], columns=GROUP_ORDER)
        .reset_index()
    )

    top_rows: list[dict[str, object]] = []
    for target, sub in ext_group.groupby("target"):
        top = sub.sort_values("mean_delta_R2", ascending=False).head(3)
        pollutant, station = split_target(target)
        row: dict[str, object] = {"target": target, "pollutant": pollutant, "station": station}
        for idx, (_, item) in enumerate(top.reset_index(drop=True).iterrows(), start=1):
            row[f"top_group_{idx}"] = item["group"]
            row[f"top_group_{idx}_delta_R2"] = float(item["mean_delta_R2"])
        top_rows.append(row)
    top_groups = pd.DataFrame(top_rows).sort_values("target")

    targeted_group = (
        targeted.groupby(["target", "removed_group"], as_index=False)
        .agg(mean_delta_R2=("r2_gain_when_included", "mean"))
    )
    targeted_group[["pollutant", "station"]] = targeted_group["target"].apply(lambda t: pd.Series(split_target(t)))

    shap_groups = shap.loc[(shap["group"] != "__feature__") & (shap["feature_set"] == "with_target_autoregressive")].copy()
    shap_groups = (
        shap_groups.groupby(["target", "group"], as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "mean"))
        .sort_values(["target", "mean_abs_shap"], ascending=[True, False])
    )
    return ext_pivot, top_groups, targeted_group, shap_groups


def build_wind_outputs(
    upwind_summary: pd.DataFrame,
    matched: pd.DataFrame,
    bootstrap: pd.DataFrame,
    threshold: pd.DataFrame,
    target_stats: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    down_up = (
        upwind_summary.loc[upwind_summary["wind_regime"].isin(["downwind", "upwind"]), ["target", "wind_regime", "mean"]]
        .pivot(index="target", columns="wind_regime", values="mean")
        .reset_index()
    )
    if {"downwind", "upwind"}.issubset(down_up.columns):
        down_up["raw_downwind_minus_upwind"] = down_up["downwind"] - down_up["upwind"]
    else:
        down_up["raw_downwind_minus_upwind"] = np.nan

    boot_pivot = bootstrap.pivot(index="target", columns="contrast", values="mean_effect").reset_index()
    ci_pivot = bootstrap.pivot(index="target", columns="contrast", values="ci95_low").reset_index()
    ci_hi_pivot = bootstrap.pivot(index="target", columns="contrast", values="ci95_high").reset_index()

    threshold_group = (
        threshold.groupby("target", as_index=False)
        .agg(
            threshold_min_diff=("downwind_minus_upwind", "min"),
            threshold_max_diff=("downwind_minus_upwind", "max"),
            threshold_mean_diff=("downwind_minus_upwind", "mean"),
        )
    )
    threshold_signs = (
        threshold.assign(sign=np.sign(threshold["downwind_minus_upwind"]))
        .groupby("target")["sign"]
        .apply(lambda s: int(s.nunique() == 1))
        .reset_index(name="threshold_sign_stable")
    )
    threshold_group = threshold_group.merge(threshold_signs, on="target", how="left")

    out = down_up[["target", "raw_downwind_minus_upwind"]].merge(
        matched[["target", "mean_diff_downwind_minus_upwind", "p_value"]].rename(
            columns={
                "mean_diff_downwind_minus_upwind": "matched_downwind_minus_upwind",
                "p_value": "matched_p_value",
            }
        ),
        on="target",
        how="left",
    )
    out = out.merge(
        boot_pivot.rename(
            columns={
                "downwind_minus_upwind": "bootstrap_downwind_minus_upwind",
                "high_downwind_minus_low_downwind": "bootstrap_high_downwind_minus_low_downwind",
            }
        ),
        on="target",
        how="left",
    )
    out = out.merge(
        ci_pivot.rename(
            columns={
                "downwind_minus_upwind": "bootstrap_downwind_minus_upwind_ci95_low",
                "high_downwind_minus_low_downwind": "bootstrap_high_downwind_minus_low_downwind_ci95_low",
            }
        ),
        on="target",
        how="left",
    )
    out = out.merge(
        ci_hi_pivot.rename(
            columns={
                "downwind_minus_upwind": "bootstrap_downwind_minus_upwind_ci95_high",
                "high_downwind_minus_low_downwind": "bootstrap_high_downwind_minus_low_downwind_ci95_high",
            }
        ),
        on="target",
        how="left",
    )
    out = out.merge(threshold_group, on="target", how="left")
    out = out.merge(target_stats, on="target", how="left")
    for col in [
        "raw_downwind_minus_upwind",
        "matched_downwind_minus_upwind",
        "bootstrap_downwind_minus_upwind",
        "bootstrap_high_downwind_minus_low_downwind",
    ]:
        out[f"{col}_std_units"] = out[col] / out["target_std"]
    out = out.sort_values("target")
    pollutant_wind = (
        out.groupby("pollutant", as_index=False)
        .agg(
            n_targets=("target", "count"),
            mean_raw_downwind_minus_upwind_std_units=("raw_downwind_minus_upwind_std_units", "mean"),
            mean_matched_downwind_minus_upwind_std_units=("matched_downwind_minus_upwind_std_units", "mean"),
            mean_bootstrap_downwind_minus_upwind_std_units=("bootstrap_downwind_minus_upwind_std_units", "mean"),
            mean_bootstrap_high_downwind_minus_low_downwind_std_units=("bootstrap_high_downwind_minus_low_downwind_std_units", "mean"),
            share_threshold_sign_stable=("threshold_sign_stable", "mean"),
        )
    )
    return out, pollutant_wind


def build_overview(
    predictability_target: pd.DataFrame,
    predictability_pollutant: pd.DataFrame,
    top_groups: pd.DataFrame,
    wind_target: pd.DataFrame,
    wind_pollutant: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_overview = predictability_target.merge(top_groups, on=["target", "pollutant", "station"], how="left")
    target_overview = target_overview.merge(wind_target, on=["target", "pollutant", "station"], how="left")
    pollutant_top = (
        top_groups.groupby("pollutant")[
            [c for c in top_groups.columns if c.startswith("top_group_") and not c.endswith("_delta_R2")]
        ]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        .reset_index()
    )
    pollutant_overview = predictability_pollutant.merge(wind_pollutant, on="pollutant", how="left")
    pollutant_overview = pollutant_overview.merge(pollutant_top, on="pollutant", how="left")
    return target_overview, pollutant_overview


def _plot_predictability_lines(predictability: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for target in [t for t in TARGET_ORDER if t in predictability["target"].unique()]:
        sub = predictability.loc[predictability["target"] == target].sort_values("horizon_h")
        ax.plot(sub["horizon_h"], sub["best_overall_R2"], marker="o", linewidth=2, label=target)
    ax.set_xlabel("hours", fontsize=12)
    ax.set_ylabel(r"$R^2$", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=14)
    fig.savefig(path, bbox_inches="tight", format=PLOT_FORMAT)
    plt.close(fig)


def _plot_heatmap(
    matrix: pd.DataFrame,
    row_col: str,
    value_cols: list[str],
    path: str,
    title: str,
    cmap: str = "coolwarm",
    center_zero: bool = True,
) -> None:
    ordered = matrix.copy()
    values = ordered[value_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.2 * len(value_cols) + 4, 0.5 * len(ordered) + 3), constrained_layout=True)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    elif center_zero:
        bound = float(np.nanmax(np.abs(finite)))
        vmin, vmax = -bound, bound
    else:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(value_cols)))
    ax.set_xticklabels(value_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered[row_col].tolist())
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(path, bbox_inches="tight", format=PLOT_FORMAT)
    plt.close(fig)


def build_plot_jobs(
    output_dir: Path,
    predictability: pd.DataFrame,
    ablation_matrix: pd.DataFrame,
    wind_summary: pd.DataFrame,
) -> list[tuple]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple] = [
        ("predictability_lines", predictability, str(plots_dir / f"cross_pollutant_best_r2_by_horizon.{PLOT_FORMAT}")),
    ]
    gain_matrix = (
        predictability.pivot(index="target", columns="horizon_h", values="autoregressive_gain_R2")
        .reindex(index=[t for t in TARGET_ORDER if t in predictability["target"].unique()])
        .reset_index()
    )
    jobs.append(
        (
            "heatmap",
            gain_matrix,
            "target",
            [c for c in gain_matrix.columns if c != "target"],
            str(plots_dir / f"cross_pollutant_autoregressive_gain.{PLOT_FORMAT}"),
            "Autoregressive gain in R2",
        )
    )
    jobs.append(
        (
            "heatmap",
            ablation_matrix,
            "target",
            [c for c in GROUP_ORDER if c in ablation_matrix.columns],
            str(plots_dir / f"cross_pollutant_ablation_heatmap.{PLOT_FORMAT}"),
            "Mean delta R2 by feature group",
        )
    )
    wind_matrix = wind_summary[
        [
            "target",
            "raw_downwind_minus_upwind_std_units",
            "matched_downwind_minus_upwind_std_units",
            "bootstrap_downwind_minus_upwind_std_units",
            "bootstrap_high_downwind_minus_low_downwind_std_units",
        ]
    ].copy()
    jobs.append(
        (
            "heatmap",
            wind_matrix,
            "target",
            [c for c in wind_matrix.columns if c != "target"],
            str(plots_dir / f"cross_pollutant_wind_response_heatmap.{PLOT_FORMAT}"),
            "Wind-regime effects in target standard deviations",
        )
    )
    return jobs


def run_plot_job(job: tuple) -> None:
    kind = job[0]
    if kind == "predictability_lines":
        _, predictability, path = job
        _plot_predictability_lines(predictability, path)
        return
    if kind == "heatmap":
        _, matrix, row_col, value_cols, path, title = job
        _plot_heatmap(matrix, row_col, value_cols, path, title)
        return
    raise ValueError(f"Tipo job sconosciuto: {kind}")


def run(args: argparse.Namespace) -> int:
    overall_start = time.perf_counter()
    if not args.no_progress:
        print("Loading inputs...", flush=True)
    frames = load_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_progress:
        print("Building target statistics...", flush=True)
    target_stats = target_std_from_dataset(frames["dataset"])

    if not args.no_progress:
        print("Building predictability summaries...", flush=True)
    predictability, predictability_target, predictability_pollutant = build_predictability_outputs(frames["summary"])
    predictability.to_csv(args.output_dir / "cross_pollutant_predictability_summary.csv", index=False)
    predictability_target.to_csv(args.output_dir / "cross_pollutant_predictability_target_overview.csv", index=False)
    predictability_pollutant.to_csv(args.output_dir / "cross_pollutant_predictability_pollutant_overview.csv", index=False)

    if not args.no_progress:
        print("Building ablation comparisons...", flush=True)
    ablation_matrix, top_groups, targeted_group, shap_groups = build_ablation_outputs(
        frames["extended_ablation"],
        frames["targeted_ablation"],
        frames["group_shap"],
    )
    ablation_matrix.to_csv(args.output_dir / "cross_pollutant_ablation_group_matrix.csv", index=False)
    top_groups.to_csv(args.output_dir / "cross_pollutant_ablation_top_groups.csv", index=False)
    targeted_group.to_csv(args.output_dir / "cross_pollutant_targeted_ablation_summary.csv", index=False)
    shap_groups.to_csv(args.output_dir / "cross_pollutant_group_shap_summary.csv", index=False)

    if not args.no_progress:
        print("Building wind-regime comparisons...", flush=True)
    wind_target, wind_pollutant = build_wind_outputs(
        frames["upwind_summary"],
        frames["matched"],
        frames["bootstrap"],
        frames["threshold"],
        target_stats,
    )
    wind_target.to_csv(args.output_dir / "cross_pollutant_wind_response_summary.csv", index=False)
    wind_pollutant.to_csv(args.output_dir / "cross_pollutant_wind_response_pollutant_overview.csv", index=False)

    if not args.no_progress:
        print("Building overview tables...", flush=True)
    target_overview, pollutant_overview = build_overview(
        predictability_target,
        predictability_pollutant,
        top_groups,
        wind_target,
        wind_pollutant,
    )
    target_overview.to_csv(args.output_dir / "cross_pollutant_overview.csv", index=False)
    pollutant_overview.to_csv(args.output_dir / "cross_pollutant_family_overview.csv", index=False)

    plot_jobs = build_plot_jobs(args.output_dir, predictability, ablation_matrix, wind_target)
    workers = resolve_workers(args.workers, len(plot_jobs))
    if not args.no_progress:
        print(f"Rendering plots with workers={workers}...", flush=True)
    if workers == 1:
        for job in plot_jobs:
            run_plot_job(job)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            list(executor.map(run_plot_job, plot_jobs))

    runtime = pd.DataFrame(
        [
            {
                "step": "total",
                "seconds": float(time.perf_counter() - overall_start),
                "workers": workers,
            }
        ]
    )
    runtime.to_csv(args.output_dir / "cross_pollutant_runtime_profile.csv", index=False)

    print(f"File scritto: {args.output_dir / 'cross_pollutant_predictability_summary.csv'}")
    print(f"File scritto: {args.output_dir / 'cross_pollutant_ablation_group_matrix.csv'}")
    print(f"File scritto: {args.output_dir / 'cross_pollutant_wind_response_summary.csv'}")
    print(f"File scritto: {args.output_dir / 'cross_pollutant_overview.csv'}")
    print(f"Cartella grafici: {args.output_dir / 'plots'}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())
