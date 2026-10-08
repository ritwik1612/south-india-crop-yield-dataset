# One master dataset

rice_master.csv is the only active merged CSV: 2,018 rows and 30 columns.
One row is a rice district-crop-year-season observation.

build_dataset.py joins original crop area/production, GADM-matched district
coordinates and NASA POWER weather; derives yield, area fraction, seasonal
weather summaries, GDD and five additional predictors. Join products are
temporary and disappear after a successful build.

The file preserves YIELD as target and PRODUCTION_TONNES as provenance.
PRODUCTION_TONNES must not be supplied to a yield predictor.
District identity, year and match status are metadata, not encoder inputs.
STATE and SEASON are contextual categorical inputs.

The 21 numeric predictors are listed in ../training/manifest.json and the
preprocessing artifact manifests. The original numeric features are retained
here even though the DAE export compresses them into latent features.
