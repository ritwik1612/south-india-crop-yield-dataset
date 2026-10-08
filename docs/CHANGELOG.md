# Changelog

All notable changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses semantic
versioning.

## [Unreleased]

## [0.2.0] - 2026-10-08

### Changed

- Streamlined active data into raw/, merged/ and training/ with readable filenames.
- Rebuilt the 2,018-row master directly from three preserved raw sources; join tables are temporary.
- Conventional and DAE representations now use the same 21 original numeric predictors.
- Executed and verified conventional preprocessing and the 180-epoch DAE run.
- Saved preprocessing artifacts, six split CSVs, predictor manifest, plots and measured findings.
- Archived superseded implementation, scripts, documents and dataset packages outside active use.
- Added exact GBDT, MLP, LSTM and 1D CNN architecture definitions; supervised yield training remains future work.
- Added the green/off-white Verdant beta with explicitly illustrative results.
- Added the complete workflow PDF, three vector diagrams, model registry and delivery checklist.
- Added a maintained Colab notebook with complete code, saved verified outputs and seven graphs.

### Added

- Added `Dataset_and_Training_Guide.md` documenting actual source provenance,
  construction rules, DAE inputs, verified cohort counts, preprocessing findings,
  training-file selection, and the distinction between implemented and executed stages.

- Added a `RAW datasets/` source-data inventory with read-only links to the Government of India crop statistics, GADM boundaries, and NASA POWER daily weather used by the pipeline.
- Added automated DL preprocessing scripts: `automated_dl_preprocessing.py` and `deep_preprocess_autoencoder.py`.
- Added deep-learning architecture and raw-versus-preprocessed comparison documentation.
- Added `MASTER_TRAINING_DATASET/` and an automated CSV-only builder that merges the training parameters into one source-backed rice-yield table.
- Documented the preprocessing model as the established Denoising Autoencoder (Vincent et al., ICML 2008), rather than a newly proposed architecture.
- Aligned automated DL preprocessing with the consolidated master training dataset and documented its 21 non-leaking DAE inputs, encoded contextual inputs, target, and exclusions.

### Planned

- Geospatial tiling with coordinate-preserving output.
- Mixed RGB/multispectral fusion experiments.
- Experiment tracking integrations and deployment profiles.

## [0.1.0] - 2026-08-06

### Added

- Initial Python package and command-line interface.
- ResNet/MobileNet classification and U-Net segmentation pipelines.
- Classification and paired-mask datasets with deterministic splitting.
- NDVI generation for red and near-infrared raster bands.
- Dataset indexing and validation commands.
- Training checkpoints, JSON history, and inference commands.
- Unit tests, sample configurations, architecture notes, and dataset guidance.
# 2026-09-24

- Added `South_India_Preprocessing_Report.tex`: a Computer Modern academic LaTeX report covering data provenance, files, features, preprocessing methods, outputs, quality findings, and modelling implications.
- Compiled and visually verified `South_India_Preprocessing_Report.pdf` (four A4 pages).
