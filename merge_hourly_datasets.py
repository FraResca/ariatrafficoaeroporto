#!/usr/bin/env python3
"""Merge aggregated datasets on hourly timestamps."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("Datasets_Raw")
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "hourly_merged_2023_2025.csv"
DEFAULT_BLQ = DEFAULT_INPUT_DIR / "BLQ_traffic_2023_2025.csv"
DEFAULT_SPIRE = DEFAULT_INPUT_DIR / "spire_flow_2023_2025.csv"
DEFAULT_POLLUTANTS = DEFAULT_INPUT_DIR / "porta_san_felice_pollutants_hourly.csv"
DEFAULT_METEO = DEFAULT_INPUT_DIR / "meteo_aero_centro_2023_2025_h.csv"
HOUR_COLUMN = "datetime"
SERVICE_TYPE_CODES = {
    "J": "scheduled_passenger",
    "P": "charter_passenger",
    "F": "cargo",
    "H": "mail",
    "C": "combined",
    "O": "other",
    "T": "technical",
    "X": "generic",
}
# Coordinates are approximate reference points used only to choose detector loops.
# For the traffic block we keep:
# - the 5 closest loops to Bologna Airport (BLQ)
# - the 5 closest loops to each pollutant station
# To avoid duplicating the exact same spira under multiple groups, selection is done
# in anchor order and already-assigned loops are skipped by later anchors.
AIRPORT_REFERENCE = (44.5354, 11.2887)
PORTA_SAN_FELICE_REFERENCE = (44.5013, 11.3280)
GIARDINI_MARGHERITA_REFERENCE = (44.4849, 11.3546)
VIA_CHIARINI_REFERENCE = (44.4946, 11.3768)
SPIRE_SELECTION_ANCHORS = {
    "airport": AIRPORT_REFERENCE,
    "porta_san_felice": PORTA_SAN_FELICE_REFERENCE,
    "giardini_margherita": GIARDINI_MARGHERITA_REFERENCE,
    "via_chiarini": VIA_CHIARINI_REFERENCE,
}
SPIRES_PER_AREA = 5
SPIRE_HOURLY_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unisce BLQ, spire, inquinanti e meteo in un unico dataset "
            "con una riga per ora."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Cartella base dei dataset (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--blq",
        type=Path,
        default=None,
        help=f"CSV traffico BLQ aggregato (default: {DEFAULT_BLQ}).",
    )
    parser.add_argument(
        "--spire",
        type=Path,
        default=None,
        help=f"CSV flussi spire aggregato (default: {DEFAULT_SPIRE}).",
    )
    parser.add_argument(
        "--pollutants",
        type=Path,
        default=None,
        help=f"CSV inquinanti orari aggregato (default: {DEFAULT_POLLUTANTS}).",
    )
    parser.add_argument(
        "--meteo",
        type=Path,
        default=None,
        help=f"CSV meteo aggregato (default: {DEFAULT_METEO}).",
    )
    parser.add_argument(
        "--flight-time-column",
        choices=["BLOCK_TIME", "SCHEDULED_TIME"],
        default="BLOCK_TIME",
        help="Colonna temporale BLQ da usare per aggregare i voli (default: BLOCK_TIME).",
    )
    parser.add_argument(
        "--join",
        choices=["outer", "inner"],
        default="outer",
        help="Tipo di merge tra dataset orari (default: outer).",
    )
    parser.add_argument(
        "--date-window",
        choices=["overlap", "all"],
        default="overlap",
        help=(
            "Periodo finale: overlap usa l'intersezione temporale dei dataset, "
            "all mantiene tutto il range disponibile (default: overlap)."
        ),
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


def resolve_default(path: Path | None, input_dir: Path, file_name: str) -> Path:
    return path if path is not None else input_dir / file_name


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    return pd.read_csv(path)


def aggregate_blq(path: Path, time_column: str) -> pd.DataFrame:
    df = read_required_csv(path)
    if time_column not in df.columns:
        raise ValueError(f"Colonna {time_column!r} mancante in {path.name}")

    df[HOUR_COLUMN] = pd.to_datetime(df[time_column], errors="coerce").dt.floor("h")
    if df[HOUR_COLUMN].isna().any():
        raise ValueError(f"Date non valide nella colonna {time_column} di {path.name}")

    for column in ["PAX_ON_BOARD", "PAX_ON_BOARD_NO_TRANSIT", "BAG_NUMBER"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
        else:
            df[column] = 0

    if "SERVICE_TYPE_CODE" in df.columns:
        service_type = (
            df["SERVICE_TYPE_CODE"]
            .astype("string")
            .str.strip()
            .str.upper()
            .fillna("X")
        )
        # Keep the official codes used by the BLQ extraction. Anything unexpected is
        # collapsed into X so the merged dataset has a stable schema across years.
        df["SERVICE_TYPE_CODE_NORMALIZED"] = service_type.where(
            service_type.isin(SERVICE_TYPE_CODES),
            "X",
        )
    else:
        df["SERVICE_TYPE_CODE_NORMALIZED"] = "X"

    grouped = (
        df.groupby(HOUR_COLUMN, as_index=False)
        .agg(
            blq_flights=("FLIGHT_NUMBER", "count"),
            blq_departures=("BOUND", lambda values: (values == "D").sum()),
            blq_arrivals=("BOUND", lambda values: (values == "A").sum()),
            blq_pax_on_board=("PAX_ON_BOARD", "sum"),
            blq_pax_no_transit=("PAX_ON_BOARD_NO_TRANSIT", "sum"),
            blq_bags=("BAG_NUMBER", "sum"),
        )
        .sort_values(HOUR_COLUMN)
    )

    service_counts = (
        df.groupby([HOUR_COLUMN, "SERVICE_TYPE_CODE_NORMALIZED"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(SERVICE_TYPE_CODES), fill_value=0)
        .rename(
            columns={
                code: f"blq_service_{label}_flights"
                for code, label in SERVICE_TYPE_CODES.items()
            }
        )
        .reset_index()
    )
    grouped = grouped.merge(service_counts, on=HOUR_COLUMN, how="left")
    return grouped


def aggregate_spire(path: Path) -> pd.DataFrame:
    df = read_required_csv(path)
    metadata_columns = [
        "id_uni",
        "codice spira",
        "Nome via",
        "longitudine",
        "latitudine",
    ]
    missing_columns = sorted(set(["data", *metadata_columns, *SPIRE_HOURLY_COLUMNS]) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Colonne mancanti in {path.name}: {missing_columns}")

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    if df["data"].isna().any():
        raise ValueError(f"Date non valide nella colonna data di {path.name}")

    for column in SPIRE_HOURLY_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    selected = select_spires(df)
    print_selected_spires(selected)

    selected_ids = selected["id_uni"].unique().tolist()
    wide = df.loc[df["id_uni"].isin(selected_ids), ["data", "id_uni", *SPIRE_HOURLY_COLUMNS]].copy()
    long_df = wide.melt(
        id_vars=["data", "id_uni"],
        value_vars=SPIRE_HOURLY_COLUMNS,
        var_name="hour_slot",
        value_name="flow",
    )
    hour_map = {column: idx for idx, column in enumerate(SPIRE_HOURLY_COLUMNS)}
    long_df[HOUR_COLUMN] = long_df["data"] + pd.to_timedelta(long_df["hour_slot"].map(hour_map), unit="h")
    column_map = selected.set_index("id_uni")["output_column"]
    long_df["output_column"] = long_df["id_uni"].map(column_map)
    spire_hourly = (
        long_df.pivot_table(
            index=HOUR_COLUMN,
            columns="output_column",
            values="flow",
            aggfunc="sum",
        )
        .reset_index()
        .sort_values(HOUR_COLUMN)
    )
    spire_hourly.columns.name = None
    return spire_hourly


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def initial_bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = (
        math.cos(phi1) * math.sin(phi2)
        - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    )
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def signed_angle_diff_degrees(direction: pd.Series, target_direction: float) -> pd.Series:
    return (direction - target_direction + 180) % 360 - 180


def add_airport_to_psf_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    airport_to_psf_bearing = initial_bearing_degrees(
        AIRPORT_REFERENCE[0],
        AIRPORT_REFERENCE[1],
        PORTA_SAN_FELICE_REFERENCE[0],
        PORTA_SAN_FELICE_REFERENCE[1],
    )

    # W_VEC_DIR is treated as meteorological wind direction: where wind comes from.
    # To estimate transport from BLQ toward Porta San Felice, convert it to the
    # direction the air mass moves toward, then compare it with the BLQ->PSF bearing.
    for station in ["aero", "centro"]:
        direction_col = f"W_VEC_DIR_{station}"
        intensity_col = f"W_VEC_INT_{station}"
        if direction_col not in out.columns or intensity_col not in out.columns:
            continue

        wind_to_direction = (out[direction_col] + 180) % 360
        angle_diff = signed_angle_diff_degrees(wind_to_direction, airport_to_psf_bearing)
        alignment = np.cos(np.deg2rad(angle_diff))
        crosswind = np.sin(np.deg2rad(angle_diff))

        out[f"airport_to_psf_wind_alignment_{station}"] = alignment
        out[f"airport_to_psf_wind_component_{station}"] = out[intensity_col] * alignment
        out[f"airport_to_psf_crosswind_component_{station}"] = out[intensity_col] * crosswind
        out[f"airport_to_psf_wind_favorable_{station}"] = (
            alignment >= math.cos(math.radians(45))
        ).astype(int)

    return out


def safe_column_token(value: object) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return token[:32] or "spire"


def select_spires(df: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        "id_uni",
        "codice spira",
        "Nome via",
        "longitudine",
        "latitudine",
    ]
    meta = (
        df.groupby(metadata_columns, as_index=False)[SPIRE_HOURLY_COLUMNS]
        .mean()
        .copy()
    )
    meta["avg_daily_flow"] = meta[SPIRE_HOURLY_COLUMNS].sum(axis=1)
    meta["dist_airport_km"] = [
        haversine_km(AIRPORT_REFERENCE[0], AIRPORT_REFERENCE[1], lat, lon)
        for lat, lon in zip(meta["latitudine"], meta["longitudine"])
    ]
    for area, (anchor_lat, anchor_lon) in SPIRE_SELECTION_ANCHORS.items():
        meta[f"dist_{area}_km"] = [
            haversine_km(anchor_lat, anchor_lon, lat, lon)
            for lat, lon in zip(meta["latitudine"], meta["longitudine"])
        ]

    assigned_ids: set[int] = set()
    selected_frames: list[pd.DataFrame] = []
    for area in SPIRE_SELECTION_ANCHORS:
        distance_col = f"dist_{area}_km"
        candidates = meta.loc[~meta["id_uni"].isin(assigned_ids)].copy()
        chosen = (
            candidates.sort_values([distance_col, "avg_daily_flow"], ascending=[True, False])
            .head(SPIRES_PER_AREA)
            .copy()
        )
        chosen["area"] = area
        chosen["rank"] = range(1, len(chosen) + 1)
        selected_frames.append(chosen)
        assigned_ids.update(int(value) for value in chosen["id_uni"].tolist())

    selected = pd.concat(selected_frames, ignore_index=True)
    selected["output_column"] = selected.apply(
        lambda row: (
            f"spire_{row['area']}_{int(row['rank'])}_id_{int(row['id_uni'])}_"
            f"{safe_column_token(row['Nome via'])}_flow"
        ),
        axis=1,
    )
    return selected[
        [
            "area",
            "rank",
            "id_uni",
            "codice spira",
            "Nome via",
            "latitudine",
            "longitudine",
            *[f"dist_{area}_km" for area in SPIRE_SELECTION_ANCHORS],
            "avg_daily_flow",
            "output_column",
        ]
    ]


def print_selected_spires(selected: pd.DataFrame) -> None:
    print("\nSpire selezionate")
    print(
        selected[
            [
                "area",
                "rank",
                "id_uni",
                "Nome via",
                "dist_airport_km",
                *[f"dist_{area}_km" for area in SPIRE_SELECTION_ANCHORS if area != "airport"],
                "avg_daily_flow",
                "output_column",
            ]
        ].to_string(index=False)
    )

def read_pollutants(path: Path) -> pd.DataFrame:
    df = read_required_csv(path)
    if "data" not in df.columns:
        raise ValueError(f"Colonna 'data' mancante in {path.name}")

    df = df.rename(columns={"data": HOUR_COLUMN})
    df[HOUR_COLUMN] = pd.to_datetime(df[HOUR_COLUMN], errors="coerce").dt.floor("h")
    if df[HOUR_COLUMN].isna().any():
        raise ValueError(f"Date non valide nella colonna data di {path.name}")
    return df.sort_values(HOUR_COLUMN)


def read_meteo(path: Path) -> pd.DataFrame:
    df = read_required_csv(path)
    if "PragaTime" not in df.columns:
        raise ValueError(f"Colonna 'PragaTime' mancante in {path.name}")

    df = df.rename(columns={"PragaTime": HOUR_COLUMN})
    df[HOUR_COLUMN] = pd.to_datetime(df[HOUR_COLUMN], errors="coerce").dt.floor("h")
    if df[HOUR_COLUMN].isna().any():
        raise ValueError(f"Date non valide nella colonna PragaTime di {path.name}")
    return df.sort_values(HOUR_COLUMN)


def print_summary(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print(f"Righe: {len(df):,}")
    print(f"{HOUR_COLUMN}: {df[HOUR_COLUMN].min()} -> {df[HOUR_COLUMN].max()}")
    print(f"Duplicati orari: {df.duplicated([HOUR_COLUMN]).sum():,}")
    value_columns = [column for column in df.columns if column != HOUR_COLUMN]
    if value_columns:
        print(f"Colonne dati: {len(value_columns):,}")


def merge_frames(frames: list[pd.DataFrame], how: str) -> pd.DataFrame:
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=HOUR_COLUMN, how=how)
    return merged.sort_values(HOUR_COLUMN)


def overlap_window(frames: list[pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts = [frame[HOUR_COLUMN].min() for frame in frames]
    ends = [frame[HOUR_COLUMN].max() for frame in frames]
    return max(starts), min(ends)


def filter_date_window(
    merged: pd.DataFrame,
    frames: list[pd.DataFrame],
    date_window: str,
) -> pd.DataFrame:
    if date_window == "all":
        return merged

    start, end = overlap_window(frames)
    return merged.loc[
        (merged[HOUR_COLUMN] >= start) & (merged[HOUR_COLUMN] <= end)
    ].copy()


def fill_hourly_absences(df: pd.DataFrame) -> pd.DataFrame:
    blq_columns = [
        "blq_flights",
        "blq_departures",
        "blq_arrivals",
        "blq_pax_on_board",
        "blq_pax_no_transit",
        "blq_bags",
    ]
    present_blq_columns = [column for column in blq_columns if column in df.columns]
    df[present_blq_columns] = df[present_blq_columns].fillna(0)
    return df


def main() -> int:
    args = parse_args()
    blq_path = resolve_default(args.blq, args.input_dir, DEFAULT_BLQ.name)
    spire_path = resolve_default(args.spire, args.input_dir, DEFAULT_SPIRE.name)
    pollutants_path = resolve_default(args.pollutants, args.input_dir, DEFAULT_POLLUTANTS.name)
    meteo_path = resolve_default(args.meteo, args.input_dir, DEFAULT_METEO.name)

    datasets = [
        ("Traffico BLQ", aggregate_blq(blq_path, args.flight_time_column)),
        ("Flussi spire", aggregate_spire(spire_path)),
        ("Inquinanti PORTA SAN FELICE", read_pollutants(pollutants_path)),
        ("Meteo aero/centro", read_meteo(meteo_path)),
    ]

    for title, frame in datasets:
        print_summary(frame, title)

    frames = [frame for _, frame in datasets]
    merged = merge_frames(frames, args.join)
    merged = filter_date_window(merged, frames, args.date_window)
    merged = fill_hourly_absences(merged)
    merged = add_airport_to_psf_wind_features(merged)
    print_summary(merged, "Dataset finale")
    print(f"Join: {args.join}")
    print(f"Finestra date: {args.date_window}")
    print(f"Colonne: {', '.join(merged.columns)}")

    if args.dry_run:
        print("\nDry run completato: nessun file scritto.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"\nFile scritto: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
