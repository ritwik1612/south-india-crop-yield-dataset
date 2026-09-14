"""Colab-ready preprocessing and EDA for the South India crop-yield dataset.

Run in Google Colab:
    !pip -q install pandas numpy matplotlib seaborn scikit-learn
    !git clone https://github.com/ritwik1612/south-india-crop-yield-dataset.git
    !python preprocess_south_india_dataset.py \
        --data-dir /content/south-india-crop-yield-dataset \
        --output-dir /content/south_india_preprocessed

If --data-dir is omitted, the repository is cloned automatically to /content.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REPOSITORY_URL = "https://github.com/ritwik1612/south-india-crop-yield-dataset.git"
DEFAULT_DATA_DIR = Path("/content/south-india-crop-yield-dataset")

# Windows are practical crop-calendar approximations for a district-level,
# season-labelled dataset. Change them only with a cited crop-calendar source.
SEASON_MONTHS = {
    "Kharif": {6, 7, 8, 9, 10, 11},
    "Rabi": {1, 2, 3, 4, 11, 12},
    "Summer": {3, 4, 5, 6},
    "Autumn": {9, 10, 11, 12},
    "Winter": {12, 1, 2, 3},
    "Whole Year": set(range(1, 13)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess and visualise South India rice-yield data.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Cloned dataset repository directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("/content/south_india_preprocessed"))
    parser.add_argument("--repo-url", default=REPOSITORY_URL)
    parser.add_argument("--minimum-weather-days", type=int, default=60)
    return parser.parse_args()


def ensure_dataset(data_dir: Path | None, repo_url: str) -> Path:
    """Return a local dataset directory, cloning the public repository if needed."""
    resolved = data_dir or DEFAULT_DATA_DIR
    if (resolved / "YIELD_NUTS2_SOUTH_INDIA_RICE.csv").exists():
        return resolved
    if resolved.exists() and any(resolved.iterdir()):
        raise FileNotFoundError(f"{resolved} exists but does not contain the dataset CSV files.")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(resolved)], check=True)
    return resolved


def seasonal_weather(labels: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily NASA POWER data into label-specific crop-season features."""
    labels = labels.copy()
    labels["FYEAR"] = labels["FYEAR"].astype(int)
    weather = weather.copy()
    weather["DATE"] = pd.to_datetime(weather["DATE"], errors="coerce")
    weather = weather.dropna(subset=["DATE"])
    weather["WEATHER_YEAR"] = weather["DATE"].dt.year
    weather["MONTH"] = weather["DATE"].dt.month
    numeric_weather = ["TMAX", "TMIN", "TAVG", "VPRES", "WSPD", "PREC", "ET0", "RAD", "RELH", "PS"]
    weather[numeric_weather] = weather[numeric_weather].apply(pd.to_numeric, errors="coerce")
    weather["GDD_BASE10"] = (weather["TAVG"] - 10).clip(lower=0)

    calendar_rows = []
    for row in labels[["IDREGION", "FYEAR", "SEASON"]].drop_duplicates().itertuples(index=False):
        months = SEASON_MONTHS.get(row.SEASON)
        if months is None:
            continue
        for month in months:
            # Rabi/Winter seasons start in Nov/Dec of the prior calendar year.
            weather_year = row.FYEAR - 1 if row.SEASON in {"Rabi", "Winter"} and month in {11, 12} else row.FYEAR
            calendar_rows.append((row.IDREGION, row.FYEAR, row.SEASON, weather_year, month))
    calendar = pd.DataFrame(
        calendar_rows,
        columns=["IDREGION", "FYEAR", "SEASON", "WEATHER_YEAR", "MONTH"],
    )
    selected_days = calendar.merge(weather, on=["IDREGION", "WEATHER_YEAR", "MONTH"], how="inner")
    aggregations = {
        "DATE": "count", "TMAX": "mean", "TMIN": "mean", "TAVG": "mean", "VPRES": "mean",
        "WSPD": "mean", "PREC": "sum", "ET0": "sum", "RAD": "sum", "RELH": "mean", "PS": "mean",
        "GDD_BASE10": "sum",
    }
    output = selected_days.groupby(["IDREGION", "FYEAR", "SEASON"], as_index=False).agg(aggregations)
    return output.rename(
        columns={
            "DATE": "WEATHER_DAY_COUNT", "TMAX": "TMAX_MEAN", "TMIN": "TMIN_MEAN", "TAVG": "TAVG_MEAN",
            "VPRES": "VPRES_MEAN", "WSPD": "WSPD_MEAN", "PREC": "PREC_TOTAL", "ET0": "ET0_TOTAL",
            "RAD": "RAD_TOTAL", "RELH": "RELH_MEAN", "PS": "PS_MEAN",
        }
    )


def make_graphs(data: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    plt.figure(figsize=(9, 5))
    sns.countplot(data=data, x="STATE", hue="SEASON", order=sorted(data["STATE"].unique()))
    plt.title("Model-ready rice-yield observations by state and season")
    plt.xlabel("State")
    plt.ylabel("Observations")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "01_observation_coverage.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=data, x="STATE", y="YIELD", order=sorted(data["STATE"].unique()))
    plt.title("Rice-yield distribution by state")
    plt.xlabel("State")
    plt.ylabel("Yield (tonnes/hectare)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "02_yield_distribution.png", dpi=200)
    plt.close()

    annual = data.groupby(["FYEAR", "STATE"], as_index=False)["YIELD"].mean()
    plt.figure(figsize=(11, 5))
    sns.lineplot(data=annual, x="FYEAR", y="YIELD", hue="STATE", marker="o")
    plt.title("Mean rice yield over time")
    plt.xlabel("Crop year")
    plt.ylabel("Mean yield (tonnes/hectare)")
    plt.tight_layout()
    plt.savefig(output_dir / "03_annual_yield_by_state.png", dpi=200)
    plt.close()

    correlation_columns = ["YIELD", "TAVG_MEAN", "PREC_TOTAL", "RAD_TOTAL", "RELH_MEAN", "GDD_BASE10", "CROP_AREA"]
    correlation = data[correlation_columns].corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, cmap="vlag", center=0, annot=True, fmt=".2f", square=True)
    plt.title("Numeric-feature correlation matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "04_feature_correlation.png", dpi=200)
    plt.close()


def chronological_split(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Keep newest years fully held out to prevent temporal leakage."""
    years = sorted(data["FYEAR"].unique())
    if len(years) < 7:
        raise ValueError("At least seven crop years are required for chronological splitting.")
    test_years = years[-2:]
    validation_years = years[-4:-2]
    return {
        "train": data.loc[~data["FYEAR"].isin(test_years + validation_years)].copy(),
        "validation": data.loc[data["FYEAR"].isin(validation_years)].copy(),
        "test": data.loc[data["FYEAR"].isin(test_years)].copy(),
    }


def dense_one_hot_encoder() -> OneHotEncoder:
    """Support both current Colab and older scikit-learn releases."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ensure_dataset(args.data_dir, args.repo_url)

    labels = pd.read_csv(data_dir / "YIELD_NUTS2_SOUTH_INDIA_RICE.csv")
    areas = pd.read_csv(data_dir / "CROP_AREA_NUTS2_SOUTH_INDIA_RICE.csv")
    centroids = pd.read_csv(data_dir / "CENTROIDS_NUTS2_SOUTH_INDIA.csv")
    weather = pd.read_csv(data_dir / "METEO_DAILY_NUTS2_SOUTH_INDIA.csv")

    labels["FYEAR"] = labels["FYEAR"].astype(int)
    labels["YIELD"] = pd.to_numeric(labels["YIELD"], errors="coerce")
    labels = labels.dropna(subset=["YIELD"])
    # Only the spatially verified cohort can be joined to observed weather.
    labels = labels.merge(centroids[["IDREGION", "STATE", "DISTRICT", "CENTROID_X", "CENTROID_Y"]], on="IDREGION", how="inner")
    labels = labels.merge(areas[["IDREGION", "FYEAR", "SEASON", "CROP_AREA"]], on=["IDREGION", "FYEAR", "SEASON"], how="left")

    weather_features = seasonal_weather(labels, weather)
    model_data = labels.merge(weather_features, on=["IDREGION", "FYEAR", "SEASON"], how="inner")
    model_data = model_data.loc[model_data["WEATHER_DAY_COUNT"] >= args.minimum_weather_days].copy()
    model_data = model_data.sort_values(["FYEAR", "STATE", "DISTRICT", "SEASON"]).reset_index(drop=True)
    model_data.to_csv(output_dir / "rice_yield_model_ready_raw.csv", index=False)
    make_graphs(model_data, output_dir)

    numeric_features = [
        "CROP_AREA", "CENTROID_X", "CENTROID_Y", "WEATHER_DAY_COUNT", "TMAX_MEAN", "TMIN_MEAN", "TAVG_MEAN",
        "VPRES_MEAN", "WSPD_MEAN", "PREC_TOTAL", "ET0_TOTAL", "RAD_TOTAL", "RELH_MEAN", "PS_MEAN", "GDD_BASE10",
    ]
    categorical_features = ["STATE", "SEASON"]
    metadata_columns = ["IDREGION", "STATE", "DISTRICT", "FYEAR", "SEASON", "YIELD"]
    splits = chronological_split(model_data)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", dense_one_hot_encoder())]), categorical_features),
        ],
        remainder="drop",
    )
    train_features = preprocessor.fit_transform(splits["train"][numeric_features + categorical_features])
    feature_names = preprocessor.get_feature_names_out().tolist()
    for split_name, split_data in splits.items():
        features = train_features if split_name == "train" else preprocessor.transform(split_data[numeric_features + categorical_features])
        processed = pd.concat(
            [split_data[metadata_columns].reset_index(drop=True), pd.DataFrame(features, columns=feature_names)], axis=1,
        )
        processed.to_csv(output_dir / f"rice_yield_{split_name}_processed.csv", index=False)
    # Standard-library pickle keeps the pipeline usable in Colab without a
    # separate joblib dependency. Load with pickle.load(open(..., "rb")).
    with (output_dir / "rice_yield_preprocessor.pkl").open("wb") as handle:
        pickle.dump(preprocessor, handle)

    report = {
        "repository": args.repo_url,
        "input_dataset_directory": str(data_dir),
        "model_ready_rows": int(len(model_data)),
        "states": sorted(model_data["STATE"].unique().tolist()),
        "crop_year_range": [int(model_data["FYEAR"].min()), int(model_data["FYEAR"].max())],
        "split_rows": {name: int(len(frame)) for name, frame in splits.items()},
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": "YIELD",
        "minimum_weather_days": args.minimum_weather_days,
        "weather_source": "NASA POWER daily data aggregated using documented season-month windows.",
    }
    (output_dir / "preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nCreated preprocessing tables and graphs in: {output_dir}")


if __name__ == "__main__":
    main()
