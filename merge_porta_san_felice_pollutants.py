#!/usr/bin/env python3
"""Merge hourly pollutant measurements for selected Bologna stations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


DEFAULT_INPUT_DIR = Path("Datasets_Raw")
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "porta_san_felice_pollutants_hourly.csv"
DEFAULT_STATIONS = ["PORTA SAN FELICE", "GIARDINI MARGHERITA", "VIA CHIARINI"]
PRIMARY_STATION = "PORTA SAN FELICE"
DATETIME_COLUMN = "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unisce gli Export_BO orari per le stazioni selezionate, "
            "producendo una colonna per inquinante/stazione."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Cartella con gli Export_BO_*.csv (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV di output (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        default=DEFAULT_STATIONS,
        help=f"Stazioni da includere (default: {'; '.join(DEFAULT_STATIONS)}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida e mostra il riepilogo senza scrivere il CSV.",
    )
    return parser.parse_args()


def pollutant_from_filename(path: Path) -> str:
    return path.stem.removeprefix("Export_BO_")


def station_token(station: str) -> str:
    return station.lower().replace(" ", "_")


def output_column_name(pollutant: str, station: str) -> str:
    return f"{pollutant}_{station_token(station)}"


def is_hourly(datetime_series: pd.Series) -> bool:
    hours = set(datetime_series.dt.hour.dropna().unique())
    return len(hours) > 1


def read_pollutant(path: Path, stations: list[str]) -> tuple[pd.DataFrame | None, str | None]:
    df = pd.read_csv(path)
    if DATETIME_COLUMN not in df.columns:
        return None, f"saltato: colonna {DATETIME_COLUMN!r} mancante"

    present_stations = [station for station in stations if station in df.columns]
    if not present_stations:
        return None, "saltato: nessuna stazione richiesta presente"

    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], errors="coerce")
    if df[DATETIME_COLUMN].isna().any():
        return None, "saltato: date non valide"

    if not is_hourly(df[DATETIME_COLUMN]):
        return None, "saltato: misurazione non oraria"

    pollutant = pollutant_from_filename(path)
    out = pd.DataFrame({DATETIME_COLUMN: df[DATETIME_COLUMN]})
    for station in present_stations:
        out[output_column_name(pollutant, station)] = pd.to_numeric(df[station], errors="coerce")
        # Keep legacy unsuffixed Porta San Felice columns for old scripts/results.
        # New analyses use the explicit *_porta_san_felice names.
        if station == PRIMARY_STATION:
            out[pollutant] = out[output_column_name(pollutant, station)]
    return out, None


def print_frame_summary(df: pd.DataFrame, pollutant: str, file_name: str) -> None:
    print(f"\n{file_name}")
    print(f"Inquinante: {pollutant}")
    print(f"Righe: {len(df):,}")
    print(f"Data: {df[DATETIME_COLUMN].min()} -> {df[DATETIME_COLUMN].max()}")
    for column in df.columns:
        if column == DATETIME_COLUMN:
            continue
        print(f"{column}: valori non nulli {df[column].notna().sum():,}")
    print(f"Duplicati data: {df.duplicated([DATETIME_COLUMN]).sum():,}")


def print_merged_summary(df: pd.DataFrame) -> None:
    print("\nDataset unito")
    print(f"Righe: {len(df):,}")
    print(f"Data: {df[DATETIME_COLUMN].min()} -> {df[DATETIME_COLUMN].max()}")
    for column in df.columns:
        if column == DATETIME_COLUMN:
            continue
        print(f"{column}: valori non nulli {df[column].notna().sum():,}")


def main() -> int:
    args = parse_args()
    files = sorted(args.input_dir.glob("Export_BO_*.csv"))
    if not files:
        raise FileNotFoundError(f"Nessun file Export_BO_*.csv trovato in {args.input_dir}")

    frames: list[pd.DataFrame] = []
    skipped: list[tuple[str, str]] = []
    for path in files:
        frame, reason = read_pollutant(path, args.stations)
        if frame is None:
            skipped.append((path.name, reason or "saltato"))
            continue

        pollutant = pollutant_from_filename(path)
        print_frame_summary(frame, pollutant, path.name)
        frames.append(frame)

    if not frames:
        raise ValueError("Nessun Export_BO orario contiene le stazioni richieste")

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=DATETIME_COLUMN, how="outer")

    merged = merged.sort_values(DATETIME_COLUMN)
    print_merged_summary(merged)

    if skipped:
        print("\nFile saltati")
        for file_name, reason in skipped:
            print(f"{file_name}: {reason}")

    if args.dry_run:
        print("\nDry run completato: nessun file scritto.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"\nFile scritto: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
