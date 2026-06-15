#!/usr/bin/env python3
"""Train a first Decision Tree model to predict hourly NO2."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, export_text
from xgboost import XGBRegressor


DEFAULT_INPUT = Path("Datasets_Raw/hourly_merged_2023_2025.csv")
DEFAULT_OUTPUT_DIR = Path("Models")
DATETIME_COLUMN = "datetime"
TARGET_COLUMN = "NO2"
NO2_LAGS = [1, 2, 3, 6, 12, 24]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Addestra un Decision Tree per predire NO2 orario."
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
        help=f"Cartella output modello/report (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Orizzonte di previsione in ore: target NO2 a t+horizon (default: 1).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Quota finale del dataset usata come test temporale (default: 0.2).",
    )
    parser.add_argument(
        "--model",
        choices=["decision_tree", "random_forest", "xgboost"],
        default="decision_tree",
        help="Modello da addestrare (default: decision_tree).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Profondita massima del modello ad alberi (default: 6).",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=24,
        help="Minimo campioni per foglia (default: 24).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Numero di alberi per Random Forest/XGBoost (default: 300).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate per XGBoost (default: 0.05).",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Subsample righe per XGBoost (default: 0.8).",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="Subsample colonne per XGBoost (default: 0.8).",
    )
    return parser.parse_args()


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    df = pd.read_csv(path)
    if DATETIME_COLUMN not in df.columns:
        raise ValueError(f"Colonna {DATETIME_COLUMN!r} mancante")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Colonna target {TARGET_COLUMN!r} mancante")

    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], errors="coerce")
    if df[DATETIME_COLUMN].isna().any():
        raise ValueError(f"Date non valide nella colonna {DATETIME_COLUMN}")

    return df.sort_values(DATETIME_COLUMN).reset_index(drop=True)


def add_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    featured = df.copy()
    featured["hour"] = featured[DATETIME_COLUMN].dt.hour
    featured["dayofweek"] = featured[DATETIME_COLUMN].dt.dayofweek
    featured["month"] = featured[DATETIME_COLUMN].dt.month
    featured["is_weekend"] = (featured["dayofweek"] >= 5).astype(int)

    for lag in NO2_LAGS:
        featured[f"NO2_lag_{lag}h"] = featured[TARGET_COLUMN].shift(lag)

    featured["NO2_rolling_6h_mean"] = (
        featured[TARGET_COLUMN].shift(1).rolling(window=6, min_periods=3).mean()
    )
    featured["NO2_rolling_24h_mean"] = (
        featured[TARGET_COLUMN].shift(1).rolling(window=24, min_periods=12).mean()
    )
    featured["target_NO2"] = featured[TARGET_COLUMN].shift(-horizon)
    return featured


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {DATETIME_COLUMN, "target_NO2"}
    return [column for column in df.columns if column not in excluded]


def temporal_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("--test-size deve essere tra 0 e 1")

    split_idx = int(len(df) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Split temporale non valido")
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    non_zero_mask = y_true != 0
    mape = (
        ((y_true[non_zero_mask] - y_pred[non_zero_mask]).abs() / y_true[non_zero_mask])
        .mean()
        * 100
    )
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAPE_%": mape,
        "R2": r2_score(y_true, y_pred),
    }


def print_metrics(title: str, values: dict[str, float]) -> None:
    print(f"\n{title}")
    for name, value in values.items():
        print(f"{name}: {value:.4f}")


def build_regressor(
    args: argparse.Namespace,
) -> DecisionTreeRegressor | RandomForestRegressor | XGBRegressor:
    if args.model == "decision_tree":
        return DecisionTreeRegressor(
            random_state=42,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
        )

    if args.model == "random_forest":
        return RandomForestRegressor(
            random_state=42,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            n_jobs=-1,
        )

    return XGBRegressor(
        random_state=42,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_samples_leaf,
        objective="reg:squarederror",
        n_jobs=-1,
    )


def main() -> int:
    args = parse_args()
    df = add_features(read_dataset(args.input), args.horizon)
    df = df.dropna(subset=[TARGET_COLUMN, "target_NO2"])
    df = df.dropna(subset=[f"NO2_lag_{max(NO2_LAGS)}h"])
    df = df.reset_index(drop=True)

    train_df, test_df = temporal_split(df, args.test_size)
    features = feature_columns(df)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", build_regressor(args)),
        ]
    )

    X_train = train_df[features]
    y_train = train_df["target_NO2"]
    X_test = test_df[features]
    y_test = test_df["target_NO2"]

    model.fit(X_train, y_train)
    tree_pred = model.predict(X_test)
    persistence_pred = test_df[TARGET_COLUMN]

    print("Dataset")
    print(f"Righe modellabili: {len(df):,}")
    print(f"Train: {len(train_df):,} righe ({train_df[DATETIME_COLUMN].min()} -> {train_df[DATETIME_COLUMN].max()})")
    print(f"Test:  {len(test_df):,} righe ({test_df[DATETIME_COLUMN].min()} -> {test_df[DATETIME_COLUMN].max()})")
    print(f"Target: NO2 a t+{args.horizon}h")
    print(f"Modello: {args.model}")
    print(f"Feature: {len(features):,}")

    print_metrics("Baseline persistence", metrics(y_test, persistence_pred))
    model_titles = {
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    model_title = model_titles[args.model]
    print_metrics(model_title, metrics(y_test, tree_pred))

    fitted_tree = model.named_steps["model"]
    fitted_feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importances = (
        pd.DataFrame(
            {
                "feature": fitted_feature_names,
                "importance": fitted_tree.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\nTop feature")
    print(importances.head(15).to_string(index=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = test_df[[DATETIME_COLUMN, TARGET_COLUMN, "target_NO2"]].copy()
    predictions["baseline_prediction"] = persistence_pred.to_numpy()
    prediction_column = f"{args.model}_prediction"
    predictions[prediction_column] = tree_pred
    predictions_path = args.output_dir / f"no2_{args.model}_predictions.csv"
    importances_path = args.output_dir / f"no2_{args.model}_feature_importances.csv"
    tree_path = args.output_dir / f"no2_{args.model}_rules.txt"
    predictions.to_csv(predictions_path, index=False)
    importances.to_csv(importances_path, index=False)
    if args.model == "decision_tree":
        tree_path.write_text(
            export_text(fitted_tree, feature_names=list(fitted_feature_names)),
            encoding="utf-8",
        )

    print(f"\nFile scritto: {predictions_path}")
    print(f"File scritto: {importances_path}")
    if args.model == "decision_tree":
        print(f"File scritto: {tree_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
