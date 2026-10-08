"""Deep-learning preprocessing for the South India rice-yield project.

This script turns the joined model-ready table into a denoised latent-feature
dataset. It deliberately complements, rather than replaces, transparent data
cleaning: joins, quality rules and unit-aware seasonal aggregation happen first;
the autoencoder then learns a compact representation of the numeric variables.

Model basis: Denoising Autoencoder (DAE), an established unsupervised feature
learning method introduced by Vincent et al. (ICML 2008), DOI
10.1145/1390156.1390294. This implements that published model family for this
project's train-only tabular features; it is not a pretrained agriculture model.

Recommended input is the consolidated training table built from the raw crop,
boundary and NASA POWER weather sources:
    python denoising_autoencoder.py \
        --model-ready-path DATASET/merged/rice_master.csv \
        --output-dir DATASET/training/autoencoder
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CORE_NUMERIC_FEATURES = [
    "CROP_AREA", "CENTROID_X", "CENTROID_Y", "WEATHER_DAY_COUNT",
    "TMAX_MEAN", "TMIN_MEAN", "TAVG_MEAN", "VPRES_MEAN", "WSPD_MEAN",
    "PREC_TOTAL", "ET0_TOTAL", "RAD_TOTAL", "RELH_MEAN", "PS_MEAN",
    "GDD_BASE10",
]
MASTER_EXTRA_NUMERIC_FEATURES = [
    "AREA_FRACTION", "TEMP_RANGE_MEAN", "WATER_BALANCE_PROXY",
    "RAIN_PER_GDD", "RADIATION_PER_GDD", "CROP_AREA_LOG1P",
]
CATEGORICAL_FEATURES = ["STATE", "SEASON"]
METADATA_COLUMNS = ["IDREGION", "STATE", "DISTRICT", "FYEAR", "SEASON", "YIELD"]


def select_numeric_features(data: pd.DataFrame) -> list[str]:
    """Select valid non-leaking numeric inputs for either supported input table.

    `PRODUCTION_TONNES` is intentionally absent: yield is derived from
    production divided by crop area, so it would leak the target. Crop year,
    district identifiers and match-status fields remain provenance metadata.
    """
    missing_core = [feature for feature in CORE_NUMERIC_FEATURES if feature not in data.columns]
    if missing_core:
        raise ValueError(f"The input table is missing required numeric columns: {missing_core}")
    return [
        feature for feature in [*CORE_NUMERIC_FEATURES, *MASTER_EXTRA_NUMERIC_FEATURES]
        if feature in data.columns
    ]


class DenoisingAutoencoder(nn.Module):
    """A compact nonlinear encoder/decoder for standardized numeric features."""

    def __init__(self, input_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.GELU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.GELU(),
            nn.Linear(32, 64), nn.GELU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(values)
        return self.decoder(latent), latent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a denoising autoencoder for tabular preprocessing.")
    parser.add_argument("--model-ready-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=25)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chronological_split(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    years = sorted(data["FYEAR"].unique())
    if len(years) < 7:
        raise ValueError("At least seven crop years are required for chronological splitting.")
    return {
        "train": data.loc[~data["FYEAR"].isin(years[-4:])].copy(),
        "validation": data.loc[data["FYEAR"].isin(years[-4:-2])].copy(),
        "test": data.loc[data["FYEAR"].isin(years[-2:])].copy(),
    }


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_comparison_figures(
    raw_train: pd.DataFrame,
    scaled_train: np.ndarray,
    reconstructed_train: np.ndarray,
    scaler: Pipeline,
    output_dir: Path,
    history: pd.DataFrame,
    numeric_features: list[str],
) -> None:
    """Create reproducible raw-versus-DL-preprocessed comparison figures."""
    sns.set_theme(style="whitegrid", context="notebook")
    selected = [feature for feature in ["CROP_AREA", "PREC_TOTAL", "TAVG_MEAN", "RAD_TOTAL"] if feature in numeric_features]
    selected_indices = [numeric_features.index(feature) for feature in selected]
    fig, axes = plt.subplots(len(selected), 2, figsize=(12, 13), constrained_layout=True)
    fig.suptitle("Raw features versus standardized preprocessing", fontsize=15, fontweight="bold")
    for row, (feature, index) in enumerate(zip(selected, selected_indices)):
        sns.histplot(raw_train[feature], bins=30, kde=True, color="#2F6F9F", ax=axes[row, 0])
        axes[row, 0].set_title(f"{feature}: raw values")
        axes[row, 0].set_xlabel("Original unit")
        sns.histplot(scaled_train[:, index], bins=30, kde=True, color="#E8892C", ax=axes[row, 1])
        axes[row, 1].axvline(0, color="#222222", linestyle="--", linewidth=1)
        axes[row, 1].set_title(f"{feature}: standardized input")
        axes[row, 1].set_xlabel("Z-score")
    fig.savefig(output_dir / "05_raw_vs_standardized_features.png", dpi=200)
    plt.close(fig)

    actual = scaler.named_steps["scaler"].inverse_transform(scaled_train)
    reconstructed = scaler.named_steps["scaler"].inverse_transform(reconstructed_train)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, feature in zip(axes, ["TAVG_MEAN", "PREC_TOTAL"]):
        index = numeric_features.index(feature)
        axis.scatter(actual[:, index], reconstructed[:, index], alpha=0.55, s=18, color="#3A7D44")
        limits = [min(actual[:, index].min(), reconstructed[:, index].min()), max(actual[:, index].max(), reconstructed[:, index].max())]
        axis.plot(limits, limits, "--", color="#222222", linewidth=1)
        axis.set_title(f"Autoencoder reconstruction: {feature}")
        axis.set_xlabel("Standard-preprocessed value restored to original unit")
        axis.set_ylabel("Autoencoder reconstruction")
    fig.savefig(output_dir / "06_autoencoder_reconstruction.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(history["epoch"], history["train_mse"], label="Training MSE", color="#2F6F9F")
    axis.plot(history["epoch"], history["validation_mse"], label="Validation MSE", color="#E8892C")
    axis.set_title("Denoising autoencoder training history")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Reconstruction MSE")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "07_autoencoder_training_loss.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.latent_dim < 1 or args.patience < 1:
        raise ValueError("Epochs, latent dimension and patience must be positive; batch size must be >= 2.")
    torch.set_num_threads(2)
    set_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.model_ready_path).dropna(subset=["YIELD"]).copy()
    numeric_features = select_numeric_features(data)
    data[numeric_features] = data[numeric_features].replace([np.inf, -np.inf], np.nan)
    splits = chronological_split(data)
    missing_columns = splits["train"][numeric_features].isna().all()
    if missing_columns.any():
        raise ValueError(f"Entirely missing training columns: {missing_columns[missing_columns].index.tolist()}")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", one_hot_encoder()),
    ])
    categorical_transformer = ColumnTransformer(
        [("categorical", categorical_pipeline, CATEGORICAL_FEATURES)],
        remainder="drop",
    )
    train_numeric = numeric_pipeline.fit_transform(splits["train"][numeric_features])
    train_categories = categorical_transformer.fit_transform(splits["train"][CATEGORICAL_FEATURES])
    prepared_numeric = {"train": train_numeric}
    prepared_categories = {"train": train_categories}
    for name in ("validation", "test"):
        prepared_numeric[name] = numeric_pipeline.transform(splits[name][numeric_features])
        prepared_categories[name] = categorical_transformer.transform(splits[name][CATEGORICAL_FEATURES])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingAutoencoder(len(numeric_features), args.latent_dim, args.dropout).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_function = nn.MSELoss()
    train_tensor = torch.tensor(train_numeric, dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=args.batch_size, shuffle=True,
                        drop_last=len(train_tensor) % args.batch_size == 1)
    validation_tensor = torch.tensor(prepared_numeric["validation"], dtype=torch.float32, device=device)
    history_rows: list[dict[str, float]] = []
    best_loss, best_state, best_epoch, stale_epochs = float("inf"), None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for (clean_batch,) in loader:
            clean_batch = clean_batch.to(device)
            noisy_batch = clean_batch + torch.randn_like(clean_batch) * args.noise_std
            reconstructed, _ = model(noisy_batch)
            loss = loss_function(reconstructed, clean_batch)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            reconstructed_validation, _ = model(validation_tensor)
            validation_loss = float(loss_function(reconstructed_validation, validation_tensor).cpu())
        history_rows.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "validation_mse": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    if best_state is None:
        raise ValueError("Training did not produce a finite validation loss.")
    model.load_state_dict(best_state)

    history = pd.DataFrame(history_rows)
    history.to_csv(output_dir / "autoencoder_training_history.csv", index=False)
    model.eval()
    all_reconstructions: dict[str, np.ndarray] = {}
    all_latents: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, values in prepared_numeric.items():
            tensor = torch.tensor(values, dtype=torch.float32, device=device)
            reconstruction, latent = model(tensor)
            all_reconstructions[name] = reconstruction.cpu().numpy()
            all_latents[name] = latent.cpu().numpy()

    category_names = categorical_transformer.get_feature_names_out().tolist()
    for name, split in splits.items():
        latent_columns = [f"AE_LATENT_{index:02d}" for index in range(1, args.latent_dim + 1)]
        deep_features = pd.DataFrame(all_latents[name], columns=latent_columns)
        category_features = pd.DataFrame(prepared_categories[name], columns=category_names)
        reconstruction_mse = ((prepared_numeric[name] - all_reconstructions[name]) ** 2).mean(axis=1)
        result = pd.concat(
            [split[METADATA_COLUMNS].reset_index(drop=True), deep_features, category_features], axis=1,
        )
        result["AE_RECONSTRUCTION_MSE"] = reconstruction_mse
        result.to_csv(output_dir / f"rice_yield_{name}_deep_processed.csv", index=False)

    actual_train = numeric_pipeline.named_steps["scaler"].inverse_transform(train_numeric)
    reconstructed_train = numeric_pipeline.named_steps["scaler"].inverse_transform(all_reconstructions["train"])
    comparison = pd.DataFrame({
        "feature": numeric_features,
        "raw_mean": actual_train.mean(axis=0),
        "raw_std": actual_train.std(axis=0),
        "standardized_mean": train_numeric.mean(axis=0),
        "standardized_std": train_numeric.std(axis=0),
        "autoencoder_reconstruction_mae": np.abs(actual_train - reconstructed_train).mean(axis=0),
    })
    comparison.to_csv(output_dir / "raw_vs_deep_preprocessed_comparison.csv", index=False)
    make_comparison_figures(
        splits["train"], train_numeric, all_reconstructions["train"],
        numeric_pipeline, output_dir, history, numeric_features,
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "numeric_features": numeric_features,
        "latent_dim": args.latent_dim,
        "dropout": args.dropout,
        "numeric_preprocessor": numeric_pipeline,
        "categorical_preprocessor": categorical_transformer,
    }, output_dir / "rice_yield_denoising_autoencoder.pt")
    report = {
        "purpose": "Denoising autoencoder feature-learning preprocessing; not synthetic-data generation.",
        "input_rows": int(len(data)),
        "split_rows": {name: int(len(split)) for name, split in splits.items()},
        "numeric_features": numeric_features,
        "categorical_features": CATEGORICAL_FEATURES,
        "latent_dim": args.latent_dim,
        "architecture": [len(numeric_features), 64, 32, args.latent_dim, 32, 64, len(numeric_features)],
        "epochs": args.epochs,
        "epochs_completed": len(history_rows),
        "best_epoch": best_epoch,
        "best_validation_reconstruction_mse": best_loss,
        "input_path": str(args.model_ready_path),
        "predictor_columns": latent_columns + category_names + ["AE_RECONSTRUCTION_MSE"],
        "noise_std": args.noise_std,
        "dropout": args.dropout,
        "device": str(device),
    }
    (output_dir / "deep_preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

