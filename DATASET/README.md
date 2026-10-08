# Dataset navigation

Only three active stages:

1. raw/: government crop statistics, district boundaries and daily weather.
2. merged/rice_master.csv: the single 2,018-row, 30-column source-derived table.
3. training/: conventional and DAE feature representations, each split into
   train.csv, validation.csv and test.csv.

Use training/manifest.json to select the model inputs explicitly.
Train model weights with train.csv; tune with validation.csv; evaluate with
test.csv. Do not concatenate the three splits for model fitting.

Run ../pre-processing/run_pipeline.py from the project root to rebuild all
stages and preserve raw files.
