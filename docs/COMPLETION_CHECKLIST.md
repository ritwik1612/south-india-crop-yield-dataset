# Delivery audit - 2026-10-08

| Request | Deliverable / evidence |
| --- | --- |
| Explain original datasets and contents | docs/Crop_Workflow.pdf; DATASET/raw/README.md |
| One merged dataset | DATASET/merged/rice_master.csv, 2,018 x 30 |
| Raw files separate | DATASET/raw/government, boundaries, weather |
| Trainable outputs separate | DATASET/training/standard and autoencoder, each train/validation/test.csv |
| Show preprocessing inputs and excluded columns | PDF; training/manifest.json; docs/MODEL_REGISTRY.md |
| Full established preprocessing model | pre-processing/denoising_autoencoder.py; artifacts/denoising_autoencoder.pt |
| Maintained preprocessing notebook | pre-processing/Crop_Preprocessing.ipynb, visible source code and seven saved figures |
| Explain measured preprocessing findings | docs/preprocessing/*.json and figures/*.png; PDF |
| Construction, training and user-computation diagrams | Three vector diagrams in the PDF |
| Exact future model names and implementations | MODEL_REGISTRY.md; APP/models/tabular.py and temporal.py |
| Green/off-white beta without yield training | APP/prototype; explicitly illustrative results |
| Local clean structure | APP, DATASET, docs, github, OTHER, pre-processing only at root |
| GitHub clean structure | Same active project folders; archives excluded; latest publish recorded in commit history |
| README and changelog | docs/README.md; docs/CHANGELOG.md; repository README |

Executed: source-preserving rebuild, conventional transforms, 180-epoch DAE
fit, output verification, model shape checks, PDF render review and browser
prototype interaction checks. Supervised yield-model training, sequence-tensor
preparation and production inference are future work, as requested for the beta.

Superseded local material is recoverable under OTHER/archive.
Automatic approval review blocked permanent recursive deletion of obsolete
dataset folders, so archival moves were used instead.
