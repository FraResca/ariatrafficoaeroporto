#!/usr/bin/env python3
"""Create correlation matrix heatmaps for the merged hourly dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_INPUT = Path("Datasets_Raw/hourly_merged_2023_2025.csv")
DEFAULT_OUTPUT_DIR = Path("Analysis")
DATETIME_COLUMN = "datetime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcola e salva la matrice di correlazione del dataset orario unito."
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
        help=f"Cartella output per CSV e immagini (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--method",
        choices=["pearson", "spearman", "kendall"],
        default="pearson",
        help="Metodo di correlazione (default: pearson).",
    )
    parser.add_argument(
        "--target",
        default="NO2",
        help="Colonna target per il riepilogo ordinato (default: NO2).",
    )
    return parser.parse_args()


def read_numeric_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    df = pd.read_csv(path)
    if DATETIME_COLUMN in df.columns:
        df = df.drop(columns=[DATETIME_COLUMN])

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        raise ValueError("Il dataset non contiene colonne numeriche")
    return numeric_df


def save_heatmap(corr: pd.DataFrame, path: Path, title: str) -> None:
    width = max(12, len(corr.columns) * 0.55)
    height = max(10, len(corr.columns) * 0.50)
    plt.figure(figsize=(width, height))
    sns.heatmap(
        corr,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.25,
        cbar_kws={"label": "correlation"},
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_target_heatmap(corr: pd.DataFrame, target: str, path: Path) -> pd.Series:
    if target not in corr.columns:
        raise ValueError(f"Target {target!r} non presente nella matrice di correlazione")

    target_corr = corr[target].drop(index=target).sort_values(key=lambda s: s.abs(), ascending=False)
    plot_df = target_corr.to_frame(name=f"corr_with_{target}")

    height = max(8, len(plot_df) * 0.35)
    plt.figure(figsize=(6, height))
    sns.heatmap(
        plot_df,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.25,
        cbar_kws={"label": "correlation"},
    )
    plt.title(f"Correlazione con {target}")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return target_corr


def main() -> int:
    args = parse_args()
    numeric_df = read_numeric_dataset(args.input)
    corr = numeric_df.corr(method=args.method)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_dir / f"correlation_matrix_{args.method}.csv"
    heatmap_path = args.output_dir / f"correlation_heatmap_{args.method}.png"
    target_heatmap_path = args.output_dir / f"correlation_with_{args.target}_{args.method}.png"
    target_csv_path = args.output_dir / f"correlation_with_{args.target}_{args.method}.csv"

    corr.to_csv(matrix_path)
    save_heatmap(corr, heatmap_path, f"Matrice di correlazione ({args.method})")
    target_corr = save_target_heatmap(corr, args.target, target_heatmap_path)
    target_corr.to_csv(target_csv_path, header=[f"corr_with_{args.target}"])

    print("Dataset numerico")
    print(f"Righe: {len(numeric_df):,}")
    print(f"Colonne numeriche: {len(numeric_df.columns):,}")
    print(f"Metodo: {args.method}")
    print(f"\nFile scritto: {matrix_path}")
    print(f"File scritto: {heatmap_path}")
    print(f"File scritto: {target_csv_path}")
    print(f"File scritto: {target_heatmap_path}")
    print(f"\nTop correlazioni assolute con {args.target}")
    print(target_corr.head(12).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
