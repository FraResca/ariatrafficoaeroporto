#!/usr/bin/env python3
"""Merge hourly weather files for aero and centro stations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


DEFAULT_INPUT_DIR = Path("Datasets_Raw")
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "meteo_aero_centro_2023_2025_h.csv"
DATETIME_COLUMN = "PragaTime"
STATIONS = ["aero", "centro"]
YEARS = [2023, 2024, 2025]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unisce i file meteo orari aero e centro 2023-2025, "
            "aggiungendo i suffissi _aero e _centro alle colonne meteo."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Cartella con i CSV meteo di input (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV di output (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida e mostra il riepilogo senza scrivere il CSV.",
    )
    return parser.parse_args()


def expected_file(input_dir: Path, station: str, year: int) -> Path:
    return input_dir / f"meteo_{station}_{year}_h.csv"


def read_meteo_csv(path: Path, expected_columns: list[str] | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    df = pd.read_csv(path)
    columns = list(df.columns)
    if expected_columns is not None and columns != expected_columns:
        raise ValueError(
            f"Schema non compatibile in {path.name}.\n"
            f"Attese: {expected_columns}\n"
            f"Trovate: {columns}"
        )

    if DATETIME_COLUMN not in df.columns:
        raise ValueError(f"Colonna {DATETIME_COLUMN!r} mancante in {path.name}")

    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], errors="coerce")
    if df[DATETIME_COLUMN].isna().any():
        raise ValueError(f"Date non valide nella colonna {DATETIME_COLUMN} di {path.name}")

    for column in df.columns:
        if column == DATETIME_COLUMN:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def merge_station(input_dir: Path, station: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    expected_columns: list[str] | None = None

    for year in YEARS:
        path = expected_file(input_dir, station, year)
        df = read_meteo_csv(path, expected_columns)
        if expected_columns is None:
            expected_columns = list(df.columns)
        print_file_summary(df, path.name)
        frames.append(df)

    station_df = pd.concat(frames, ignore_index=True)
    duplicate_count = int(station_df.duplicated([DATETIME_COLUMN]).sum())
    if duplicate_count:
        raise ValueError(f"Trovati {duplicate_count:,} duplicati orari per meteo_{station}")

    rename_columns = {
        column: f"{column}_{station}"
        for column in station_df.columns
        if column != DATETIME_COLUMN
    }
    return station_df.rename(columns=rename_columns)


def print_file_summary(df: pd.DataFrame, file_name: str) -> None:
    print(f"\n{file_name}")
    print(f"Righe: {len(df):,}")
    print(f"{DATETIME_COLUMN}: {df[DATETIME_COLUMN].min()} -> {df[DATETIME_COLUMN].max()}")
    print(f"Duplicati orari: {df.duplicated([DATETIME_COLUMN]).sum():,}")


def print_merged_summary(df: pd.DataFrame) -> None:
    print("\nDataset unito")
    print(f"Righe: {len(df):,}")
    print(f"{DATETIME_COLUMN}: {df[DATETIME_COLUMN].min()} -> {df[DATETIME_COLUMN].max()}")
    print(f"Colonne: {', '.join(df.columns)}")
    for column in df.columns:
        if column == DATETIME_COLUMN:
            continue
        print(f"{column}: valori non nulli {df[column].notna().sum():,}")


def main() -> int:
    args = parse_args()
    station_frames = [merge_station(args.input_dir, station) for station in STATIONS]

    merged = station_frames[0]
    for frame in station_frames[1:]:
        merged = merged.merge(frame, on=DATETIME_COLUMN, how="outer")

    merged = merged.sort_values(DATETIME_COLUMN)
    print_merged_summary(merged)

    if args.dry_run:
        print("\nDry run completato: nessun file scritto.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"\nFile scritto: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
