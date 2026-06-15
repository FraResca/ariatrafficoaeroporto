#!/usr/bin/env python3
"""Merge vehicle flow loop detector CSV files for 2023, 2024 and 2025."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd


DEFAULT_INPUT_DIR = Path("Datasets_Raw")
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "spire_flow_2023_2025.csv"
EXPECTED_FILES = [
    "rilevazione-flusso-veicoli-tramite-spire-anno-2023.csv",
    "rilevazione-flusso-veicoli-tramite-spire-anno-2024.csv",
    "rilevazione-flusso-veicoli-tramite-spire-anno-2025.csv",
]
HOURLY_COLUMNS = [
    "00:00-01:00",
    "01:00-02:00",
    "02:00-03:00",
    "03:00-04:00",
    "04:00-05:00",
    "05:00-06:00",
    "06:00-07:00",
    "07:00-08:00",
    "08:00-09:00",
    "09:00-10:00",
    "10:00-11:00",
    "11:00-12:00",
    "12:00-13:00",
    "13:00-14:00",
    "14:00-15:00",
    "15:00-16:00",
    "16:00-17:00",
    "17:00-18:00",
    "18:00-19:00",
    "19:00-20:00",
    "20:00-21:00",
    "21:00-22:00",
    "22:00-23:00",
    "23:00-24:00",
]
DEDUPLICATION_KEY = [
    "data",
    "codice spira",
    "id_uni",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unisce le rilevazioni di flusso veicoli tramite spire 2023, 2024 e 2025."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Cartella con i CSV di input (default: {DEFAULT_INPUT_DIR}).",
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
        help="Rimuove eventuali duplicati sulla chiave spira-giorno prima di salvare.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida e mostra il riepilogo senza scrivere il CSV.",
    )
    return parser.parse_args()


def year_from_filename(path: Path) -> int | None:
    match = re.search(r"anno-(\d{4})", path.name)
    return int(match.group(1)) if match else None


def read_spire_csv(path: Path, expected_columns: list[str] | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    columns = list(df.columns)
    if expected_columns is not None and columns != expected_columns:
        raise ValueError(
            f"Schema non compatibile in {path.name}.\n"
            f"Attese: {expected_columns}\n"
            f"Trovate: {columns}"
        )

    if not set(HOURLY_COLUMNS).issubset(df.columns):
        missing = sorted(set(HOURLY_COLUMNS) - set(df.columns))
        raise ValueError(f"Colonne orarie mancanti in {path.name}: {missing}")

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    if df["data"].isna().any():
        raise ValueError(f"Date non valide nella colonna data di {path.name}")

    for column in HOURLY_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any():
            raise ValueError(f"Valori orari non numerici nella colonna {column} di {path.name}")

    df["SOURCE_FILE"] = path.name
    df["SOURCE_YEAR"] = year_from_filename(path)
    return df


def print_summary(df: pd.DataFrame, title: str) -> None:
    total_flow = int(df[HOURLY_COLUMNS].sum().sum())
    print(f"\n{title}")
    print(f"Righe: {len(df):,}")
    print(f"Data: {df['data'].min().date()} -> {df['data'].max().date()}")
    print(f"Spire distinte: {df['codice spira'].nunique():,}")
    print(f"Flusso totale: {total_flow:,}")
    print(f"Duplicati chiave spira-giorno: {df.duplicated(DEDUPLICATION_KEY).sum():,}")


def main() -> int:
    args = parse_args()
    files = [args.input_dir / file_name for file_name in EXPECTED_FILES]

    frames: list[pd.DataFrame] = []
    expected_columns: list[str] | None = None
    for path in files:
        df = read_spire_csv(path, expected_columns)
        if expected_columns is None:
            expected_columns = list(df.columns[:-2])
        print_summary(df, path.name)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["data", "codice spira", "id_uni"])

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
