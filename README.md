# South India Rice Yield Studio

Three source families -> one merged rice dataset -> verified training inputs.
The project also includes the fitted Denoising Autoencoder preprocessing model,
future yield-model definitions and the green/off-white Verdant beta.

Start with [the full workflow PDF](docs/Crop_Workflow.pdf),
[the navigation guide](docs/README.md), and [the exact model registry](docs/MODEL_REGISTRY.md).

## Dataset stages

| Folder | What it contains |
| --- | --- |
| DATASET/raw/government/ | crop_statistics.csv, original APY crop statistics |
| DATASET/raw/boundaries/ | india_districts.geojson.zip, GADM 4.1 district boundaries |
| DATASET/raw/weather/ | daily_weather.csv, NASA POWER API extract |
| DATASET/merged/ | rice_master.csv: 2,018 observations x 30 columns |
| DATASET/training/standard/ | train.csv, validation.csv, test.csv: 28 conventional predictors |
| DATASET/training/autoencoder/ | train.csv, validation.csv, test.csv: 16 DAE/context/error predictors |

Four states: Andhra Pradesh, Karnataka, Tamil Nadu, Telangana.
Crop: rice. Years: 2001-2019. These are source-derived records, not invented
observations. Netherlands reference observations are not included.

**Use [DATASET/training/manifest.json](DATASET/training/manifest.json) to select
predictor columns and YIELD separately.** Train 1,485 rows; validation 245;
test 288. Identifiers are metadata, and reported production is excluded to
prevent target leakage.

## Exact model names

- Preprocessing: SimpleImputer, StandardScaler, OneHotEncoder, Denoising Autoencoder.
- Tabular computation: HistGradientBoostingRegressor (GBDT), Multilayer Perceptron (MLP) regressor.
- Future sequence computation: Long Short-Term Memory (LSTM), One-dimensional CNN (1D CNN).

The DAE is fitted (180 epochs, best validation reconstruction MSE 0.07724).
The yield models have architecture definitions; they are not yet trained.
The beta uses a fixed, explicitly illustrative yield value.

## Run preprocessing or the beta

    python -m pip install -r pre-processing/requirements.txt
    python pre-processing/run_pipeline.py --epochs 180
    python pre-processing/verify_outputs.py
    python -m http.server 8765 --directory APP/prototype

Open http://localhost:8765 for the beta.

[Open the complete preprocessing notebook in Colab](https://colab.research.google.com/github/ritwik1612/south-india-crop-yield-dataset/blob/main/pre-processing/Crop_Preprocessing.ipynb).
It includes visible source code, saved tables and seven actual graphs.
Use this maintained notebook with the new paths; the older standalone notebook
used the superseded flat CSV layout.

## Code and outputs

- APP/models/: explicit future yield architectures.
- APP/prototype/: demonstration frontend.
- pre-processing/: complete automated pipeline.
- pre-processing/artifacts/: fitted DAE weights and conventional transforms.
- docs/figures/: EDA, scale comparisons and actual DAE learning curves.
- docs/preprocessing/: manifests, reconstruction comparison and verification results.
- docs/reference_papers/: the yield reference and DAE preprocessing reference.

Old root-level CSVs are replaced by the organized stages. Older versions remain
recoverable through Git history. Historical local archives are not published.
Read DATASET/raw/README.md for source links and dataset-specific license terms;
no blanket redistribution license is asserted for third-party data.
