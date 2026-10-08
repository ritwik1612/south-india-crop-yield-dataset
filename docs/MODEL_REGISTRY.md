# Exact model names and roles

| Stage | Exact name | Code / status | What it learns |
| --- | --- | --- | --- |
| Missing values | scikit-learn SimpleImputer (median) | Fitted | Training-period median per numeric predictor |
| Numeric scaling | scikit-learn StandardScaler | Fitted | Training-period mean and standard deviation |
| Categorical context | scikit-learn OneHotEncoder | Fitted | Training-period state/season vocabulary |
| Learned preprocessing | Denoising Autoencoder (DAE), adapted Vincent et al. 2008 | pre-processing/denoising_autoencoder.py; fitted checkpoint saved | Reconstruct 21 clean standardized predictors from noisy inputs; export 8 latent features |
| Tabular baseline / comparison | scikit-learn HistGradientBoostingRegressor (GBDT) | APP/models/tabular.py, make_gbdt(); defined, not fitted | Nonlinear decision rules and interactions predicting rice yield |
| Initial tabular DL computation | Multilayer Perceptron (MLP) regressor | APP/models/tabular.py, YieldMLP; defined, not fitted | Map latent/context values to yield through dense layers 64 and 32 |
| Paper-based sequence computation | Long Short-Term Memory (LSTM) regressor | APP/models/temporal.py, YieldLSTM; defined, not fitted | Dependencies over successive weather timesteps |
| Paper-based sequence comparison | One-dimensional Convolutional Neural Network (1D CNN) regressor | APP/models/temporal.py, YieldCNN; defined, not fitted | Local temporal weather patterns |

The DAE is the preprocessing neural network. MLP/GBDT are the first tabular
yield predictors. LSTM/1D CNN reproduce the reference paper's model families
using future Indian weather sequences; they are not the same input experiment
as the current seasonal-summary CSV.

No ResNet, MobileNet or U-Net is used in the active CSV-based yield workflow.
The prototype runs a fixed demonstration value, not any fitted yield predictor.
Full-season weather currently supports retrospective estimation. Early
forecasting must restrict weather and other inputs to what is known then.
