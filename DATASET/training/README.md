# These are the supervised model's input files

| Representation | Fit weights | Select settings | Final evaluation |
| --- | --- | --- | --- |
| Conventional | standard/train.csv | standard/validation.csv | standard/test.csv |
| DAE | autoencoder/train.csv | autoencoder/validation.csv | autoencoder/test.csv |

Rows: 1,485 train; 245 validation; 288 test.
Conventional: 28 predictors. DAE: 16 predictors, including optional
AE_RECONSTRUCTION_MSE. Both representations preserve YIELD and selected
metadata in addition to predictors.

Use manifest.json: select predictor_columns, and select YIELD separately as
the label. Never include every CSV column as an input feature.

Future sequence models use separately prepared weather tensors; these CSVs
are the current tabular training representations.
