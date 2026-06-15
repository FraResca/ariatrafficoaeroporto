#!/usr/bin/env python3
"""Generate zoomable HTML plots from saved temporal CV predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


DEFAULT_PREDICTIONS = Path("Analysis/slurm_full_explain/advanced_temporal_cv_predictions.csv")
DEFAULT_SUMMARY = Path("Analysis/slurm_full_explain/advanced_temporal_cv_summary.csv")
DEFAULT_OUTPUT_DIR = Path("Analysis/slurm_full_explain/interactive_plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera grafici HTML zoommabili delle serie temporali di test a partire "
            "dalle predizioni già salvate."
        )
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--selection",
        choices=["best_per_model", "best_overall"],
        default="best_per_model",
        help=(
            "best_per_model: una linea per modello, usando per ciascun modello il suo "
            "feature_set migliore all'orizzonte scelto; best_overall: una sola linea "
            "del miglior setup complessivo."
        ),
    )
    parser.add_argument(
        "--include-range-slider",
        action="store_true",
        help="Aggiunge il range slider sull'asse temporale.",
    )
    parser.add_argument(
        "--split-by",
        choices=["fold", "month"],
        default="month",
        help=(
            "fold: un file per fold intero; month: un file per ciascun mese "
            "all'interno di ogni fold."
        ),
    )
    return parser.parse_args()


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} non trovato: {path}")
    return pd.read_csv(path)


def best_feature_sets(summary: pd.DataFrame, target: str, horizon: int, selection: str) -> pd.DataFrame:
    subset = summary.loc[(summary["target"] == target) & (summary["horizon_h"] == horizon)].copy()
    if subset.empty:
        return subset
    subset = subset.sort_values(["mean_R2", "mean_MAE"], ascending=[False, True])
    if selection == "best_overall":
        return subset.head(1)
    return subset.groupby("model", as_index=False, sort=False).head(1)


def build_plot(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    target: str,
    horizon: int,
    fold: int,
    selection: str,
    include_range_slider: bool,
) -> go.Figure | None:
    best = best_feature_sets(summary, target, horizon, selection)
    if best.empty:
        return None

    target_pred = predictions.loc[
        (predictions["target"] == target)
        & (predictions["horizon_h"] == horizon)
        & (predictions["fold"] == fold)
    ].copy()
    if target_pred.empty:
        return None
    target_pred["datetime"] = pd.to_datetime(target_pred["datetime"], errors="coerce")
    target_pred = target_pred.dropna(subset=["datetime"]).sort_values("datetime")

    fig = go.Figure()

    truth = (
        target_pred[["datetime", "y_true"]]
        .drop_duplicates()
        .sort_values("datetime")
    )
    fig.add_trace(
        go.Scatter(
            x=truth["datetime"],
            y=truth["y_true"],
            mode="lines",
            name="osservato",
            line=dict(color="black", width=2),
            hovertemplate="%{x}<br>osservato=%{y:.4f}<extra></extra>",
        )
    )

    for _, row in best.iterrows():
        model = row["model"]
        feature_set = row["feature_set"]
        series = target_pred.loc[
            (target_pred["model"] == model) & (target_pred["feature_set"] == feature_set),
            ["datetime", "y_pred"],
        ].copy()
        if series.empty:
            continue
        series = series.sort_values("datetime")
        label = model if selection == "best_overall" else f"{model} | {feature_set}"
        fig.add_trace(
            go.Scatter(
                x=series["datetime"],
                y=series["y_pred"],
                mode="lines",
                name=label,
                hovertemplate="%{x}<br>pred=%{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{target} - test set t+{horizon}h - fold {fold}",
        xaxis_title="Tempo",
        yaxis_title=target,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(rangeslider_visible=include_range_slider)
    return fig


def iter_month_windows(target_pred: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    periods = target_pred["datetime"].dt.to_period("M")
    windows: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for period in sorted(periods.dropna().unique()):
        start = period.start_time
        end = period.end_time
        label = str(period)
        windows.append((start, end, label))
    return windows


def main() -> int:
    args = parse_args()
    predictions = load_csv(args.predictions, "CSV predizioni")
    summary = load_csv(args.summary, "CSV summary")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    available_targets = sorted(predictions["target"].dropna().unique())
    targets = args.targets or available_targets
    folds = sorted(predictions.loc[predictions["horizon_h"] == args.horizon, "fold"].dropna().unique())

    generated = 0
    for target in targets:
        for fold in folds:
            fold_pred = predictions.loc[
                (predictions["target"] == target)
                & (predictions["horizon_h"] == args.horizon)
                & (predictions["fold"] == fold)
            ].copy()
            if fold_pred.empty:
                continue
            fold_pred["datetime"] = pd.to_datetime(fold_pred["datetime"], errors="coerce")
            fold_pred = fold_pred.dropna(subset=["datetime"]).sort_values("datetime")
            if fold_pred.empty:
                continue

            windows = [(None, None, None)]
            if args.split_by == "month":
                windows = iter_month_windows(fold_pred)

            for start, end, label in windows:
                pred_slice = fold_pred
                suffix = f"fold_{int(fold)}"
                if start is not None and end is not None and label is not None:
                    pred_slice = fold_pred.loc[
                        (fold_pred["datetime"] >= start) & (fold_pred["datetime"] <= end)
                    ].copy()
                    suffix = f"fold_{int(fold)}_{label}"
                if pred_slice.empty:
                    continue
                fig = build_plot(
                    predictions=pred_slice,
                    summary=summary,
                    target=target,
                    horizon=args.horizon,
                    fold=int(fold),
                    selection=args.selection,
                    include_range_slider=args.include_range_slider,
                )
                if fig is None:
                    continue
                if label is not None:
                    fig.update_layout(title=f"{target} - test set t+{args.horizon}h - fold {int(fold)} - {label}")
                out_name = f"{target}_test_series_t{args.horizon}h_{suffix}_{args.selection}.html"
                out_path = args.output_dir / out_name
                fig.write_html(out_path, include_plotlyjs="cdn")
                generated += 1
                print(out_path)

    if generated == 0:
        raise SystemExit("Nessun grafico generato: controlla target/orizzonte/file di input.")
    print(f"Generati {generated} grafici in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
