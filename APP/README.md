# Yield application: implementation stage

models/tabular.py contains the GBDT baseline factory and MLP predictor.
models/temporal.py contains the LSTM and 1D CNN sequence predictor definitions.
These are architecture definitions, not trained yield models or an application UI.

Read ../DATASET/training/manifest.json to select predictors and separate YIELD.
Only train.csv fits supervised weights; validation.csv selects settings;
test.csv measures final performance.

The first application uses tabular features. Sequence models require a separate
weather-tensor preparation stage. The LSTM accepts variable-length sequences
with true lengths; the CNN accepts fixed-length, unpadded sequences.
Crop year, region IDs and production must not accidentally become predictors.

The trained preprocessing checkpoint is in ../pre-processing/artifacts/.
For deployment, resolve weather and state-season area totals before creating
predictors. Estimate yield in t/ha; production can be estimated by multiplying
predicted yield by user-supplied area.
