#!/usr/bin/env python3
"""Merge BLQ traffic extraction files for 2023, 2024 and 2025."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd


DEFAULT_INPUT_DIR = Path("Datasets_Raw")
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "BLQ_traffic_2023_2025.csv"
EXPECTED_FILES = [
    "2023 estrazione dati di traffico BLQ.xlsx",
    "2024 estrazione dati di traffico BLQ.xlsx",
    "2025 estrazione dati di traffico BLQ_fino 1007.xlsx",
]
DATETIME_COLUMNS = ["SCHEDULED_TIME", "BLOCK_TIME"]
DEDUPLICATION_KEY = [
    "SCHEDULED_TIME",
    "BLOCK_TIME",
    "BOUND",
    "FLIGHT_NUMBER",
    "ORIDES_CODE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unisce le estrazioni di traffico BLQ 2023, 2024 e 2025."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Cartella con gli Excel di input (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV di output (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Rimuove eventuali duplicati sulla chiave volo prima di salvare.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida e mostra il riepilogo senza scrivere il CSV.",
    )
    return parser.parse_args()


def year_from_filename(path: Path) -> int | None:
    match = re.match(r"(\d{4})\b", path.name)
    return int(match.group(1)) if match else None


def read_excel(path: Path, expected_columns: list[str] | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    df = pd.read_excel(path, sheet_name="Sheet1")
    columns = list(df.columns)
    if expected_columns is not None and columns != expected_columns:
        raise ValueError(
            f"Schema non compatibile in {path.name}.\n"
            f"Attese: {expected_columns}\n"
            f"Trovate: {columns}"
        )

    for column in DATETIME_COLUMNS:
        df[column] = pd.to_datetime(df[column], errors="coerce")
        if df[column].isna().any():
            raise ValueError(f"Date non valide nella colonna {column} di {path.name}")

    df["SOURCE_FILE"] = path.name
    df["SOURCE_YEAR"] = year_from_filename(path)
    return df


def print_summary(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print(f"Righe: {len(df):,}")
    print(f"SCHEDULED_TIME: {df['SCHEDULED_TIME'].min()} -> {df['SCHEDULED_TIME'].max()}")
    print(f"BLOCK_TIME:     {df['BLOCK_TIME'].min()} -> {df['BLOCK_TIME'].max()}")
    print(f"Duplicati chiave volo: {df.duplicated(DEDUPLICATION_KEY).sum():,}")
    if "SERVICE_TYPE_CODE" in df.columns:
        counts = (
            df["SERVICE_TYPE_CODE"]
            .astype("string")
            .str.strip()
            .str.upper()
            .fillna("X")
            .value_counts(dropna=False)
            .sort_index()
        )
        print("SERVICE_TYPE_CODE:")
        print(counts.to_string())


def main() -> int:
    args = parse_args()
    files = [args.input_dir / file_name for file_name in EXPECTED_FILES]

    frames: list[pd.DataFrame] = []
    expected_columns: list[str] | None = None
    for path in files:
        df = read_excel(path, expected_columns)
        if expected_columns is None:
            expected_columns = list(df.columns[:-2])
        print_summary(df, path.name)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["BLOCK_TIME", "SCHEDULED_TIME", "FLIGHT_NUMBER"])

    duplicate_count = int(merged.duplicated(DEDUPLICATION_KEY).sum())
    if duplicate_count and args.drop_duplicates:
        merged = merged.drop_duplicates(DEDUPLICATION_KEY, keep="first")
        print(f"\nRimossi {duplicate_count:,} duplicati.")

    print_summary(merged, "Dataset unito")

    if args.dry_run:
        print("\nDry run completato: nessun file scritto.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"\nFile scritto: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
