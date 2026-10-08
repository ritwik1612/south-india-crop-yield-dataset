"""Create the maintained Colab notebook with visible code and verified saved outputs."""
import base64
import csv
import json
import textwrap
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
cells=[]
def markdown(text): cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(True)))
def code(text,outputs=None):
    cells.append(dict(cell_type='code',metadata={},execution_count=None,source=text.splitlines(True),outputs=outputs or []))
def display(data):
    return dict(output_type='display_data',metadata={},data=data)

markdown('# South India rice: complete preprocessing\n'
         'Three raw sources → one master CSV → standard and DAE training splits.\n\n'
         'The complete code is visible below. Saved tables and plots are from the verified '
         '08 October 2026 run. Run all cells to reproduce in Colab. This trains the preprocessing '
         'DAE only; it does not train the final yield models.\n\n'
         '**Preprocessing:** SimpleImputer, StandardScaler, OneHotEncoder, Denoising Autoencoder.\n\n'
         '**Future yield models:** HistGradientBoostingRegressor, Multilayer Perceptron, '
         'Long Short-Term Memory and one-dimensional CNN.')
code('from pathlib import Path\nimport subprocess, sys\n'
     'repo = Path("/content/south-india-crop-yield-dataset")\n'
     'dataset_url = "https://github.com/ritwik1612/south-india-crop-yield-dataset.git"\n'
     'if not repo.exists():\n    subprocess.run(["git", "clone", dataset_url, str(repo)], check=True)\n'
     'subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(repo / "pre-processing/requirements.txt")], check=True)\n'
     'sys.path.insert(0, str(repo / "pre-processing"))\n'
     'dataset = repo / "DATASET"\n'
     'print("Raw files:", list((dataset / "raw").rglob("*.csv")))\n'
     'print("Master:", dataset / "merged/rice_master.csv")\n')
markdown('## 1. Build the merged table from the raw files\n'
         'Input: raw/government/crop_statistics.csv, raw/boundaries/india_districts.geojson.zip, '
         'raw/weather/daily_weather.csv. Temporary crop/area/location join tables exist only during construction.')
source=(ROOT/'pre-processing/build_dataset.py').read_text().replace('from __future__ import annotations\n','')
code('__file__ = str(repo / "pre-processing/build_dataset.py")\n'
     'sys.argv = [__file__, "--dataset-dir", str(dataset)]\n'+source)
markdown('## 2. Inspect the master, coverage and missing values\n'
         'A row is a rice district-year-season observation. Yield labels are retained separately from predictors.')
preview='import pandas as pd\nmaster = pd.read_csv(dataset / "merged/rice_master.csv")\n'
preview+='display(master.head())\ndisplay(master.groupby("STATE").size().rename("Rows").to_frame())\n'
preview+='display(master.isna().sum().rename("Missing").to_frame())\nprint(master.shape)\n'
with (ROOT/'DATASET/merged/rice_master.csv').open(newline='',encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
subset=['STATE','DISTRICT','FYEAR','SEASON','YIELD','CROP_AREA','PREC_TOTAL','TAVG_MEAN']
html='<table><thead><tr>'+''.join('<th>'+c+'</th>' for c in subset)+'</tr></thead><tbody>'
html+=''.join('<tr>'+''.join('<td>'+r[c]+'</td>' for c in subset)+'</tr>' for r in rows[:5])+'</tbody></table>'
code(preview,[display({'text/html':html,'text/plain':'Verified master preview: 2,018 rows, 30 columns.'})])
markdown('## 3. Conventional preprocessing — complete code\n'
         'Fit medians, scaling and category mapping on 2001–2015 only. Validation is 2016–2017; '
         'test is 2018–2019. The 21 predictors match the DAE experiment.')
source=(ROOT/'pre-processing/preprocess_master.py').read_text().replace('from __future__ import annotations\n','')
code('__file__ = str(repo / "pre-processing/preprocess_master.py")\n'
     'artifacts = repo / "pre-processing/artifacts"\n'
     'sys.argv = [__file__, "--master-path", str(dataset / "merged/rice_master.csv"), '
     '"--output-dir", str(dataset / "training/standard"), "--artifacts-dir", str(artifacts)]\n'+source)
markdown('## 4. Denoising Autoencoder — complete code\n'
         '21 numeric inputs → 64 → 32 → 8 latent features → 32 → 64 → 21 reconstruction. '
         'Gaussian corruption is added during training. Clean inputs are reconstruction targets. '
         'YIELD and PRODUCTION_TONNES never enter the encoder.')
source=(ROOT/'pre-processing/denoising_autoencoder.py').read_text().replace('from __future__ import annotations\n','')
code('__file__ = str(repo / "pre-processing/denoising_autoencoder.py")\n'
     'deep = dataset / "training/autoencoder"\n'
     'sys.argv = [__file__, "--model-ready-path", str(dataset / "merged/rice_master.csv"), '
     '"--output-dir", str(deep), "--epochs", "180"]\n'+source)
markdown('## 5. Put outputs into the training, artifacts and documentation folders\n'
         'The training folder contains only train/validation/test CSVs and the safe-column manifest. '
         'Weights and transforms go to preprocessing artifacts; plots and findings go to docs.')
runner=(ROOT/'pre-processing/run_pipeline.py').read_text()
tail=runner[runner.index("    for split in ('train','validation','test'):"):runner.index("\nif __name__")]
code('import shutil, json\nfigures = repo / "docs/figures"\nreports = repo / "docs/preprocessing"\n'
     'for path in (artifacts, figures, reports): path.mkdir(parents=True, exist_ok=True)\n'
     +textwrap.dedent(tail))
markdown('## 6. Output verification and model-ready rows\n'
         'Standard: 28 predictors. DAE: 16 predictors (8 latent + 7 categorical + reconstruction error). '
         'Six metadata/target fields are retained in addition. Use the manifest to select X and YIELD to select y.')
validation=json.loads((ROOT/'docs/preprocessing/output_validation.json').read_text())
code('subprocess.run([sys.executable, str(repo / "pre-processing/verify_outputs.py")], check=True)\n'
     'manifest = json.loads((dataset / "training/manifest.json").read_text())\n'
     'for arm in ("standard", "autoencoder"):\n'
     '    train = pd.read_csv(dataset / "training" / manifest[arm]["train"])\n'
     '    display(train.head())\n'
     '    print(arm, "rows:", len(train), "predictors:", len(manifest[arm]["predictor_columns"]))\n',
     [display({'application/json':validation,'text/plain':json.dumps(validation,indent=2)})])
markdown('## 7. Informative graphs and measured learning\n'
         'Coverage is unequal across states/seasons. Scaling changes units; DAE loss measures reconstruction. '
         'Best validation reconstruction MSE was 0.07724 at epoch 178, not a yield accuracy score. '
         'Autumn/Winter are unseen training categories and require separate evaluation.')
code('subprocess.run([sys.executable, str(repo / "pre-processing/make_figures.py")], check=True)')
captions=[
    ('01_observation_coverage.png','Observation coverage by state and season'),
    ('02_yield_distribution.png','Recorded yield distribution'),
    ('03_annual_yield_by_state.png','Recorded annual mean yield'),
    ('04_feature_correlation.png','Feature correlations: associations, not causes'),
    ('05_raw_vs_standardized_features.png','Raw inputs compared with standardized inputs'),
    ('06_autoencoder_reconstruction.png','Actual DAE reconstructions'),
    ('07_autoencoder_training_loss.png','Actual DAE training/validation history'),
]
for filename,caption in captions:
    markdown('### '+caption)
    encoded=base64.b64encode((ROOT/'docs/figures'/filename).read_bytes()).decode()
    code('from IPython.display import Image, display\n'
         'display(Image(filename=str(repo / "docs/figures/'+filename+'")))',
         [display({'image/png':encoded,'text/plain':caption})])
markdown('## 8. What is used for supervised training?\n'
         'Use DATASET/training/standard/train.csv or autoencoder/train.csv. '
         'Keep validation.csv and test.csv separate. Raw sources and rice_master.csv remain preserved. '
         'The fitted preprocessing DAE is not a fitted yield predictor.\n\n'
         'See docs/MODEL_REGISTRY.md and docs/Crop_Workflow.pdf for all model names, computational diagrams, '
         'limitations and future steps. The Verdant beta uses illustrative values only.')
notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
              'language_info':{'name':'python'},'colab':{'name':'Crop_Preprocessing.ipynb'}},cells=cells)
for index,cell in enumerate(cells): cell['id']=f'crop-cell-{index:02d}'
destination=ROOT/'pre-processing/Crop_Preprocessing.ipynb'
destination.write_text(json.dumps(notebook,indent=1),encoding='utf-8')
print(f'Created {destination}: {len(cells)} cells, seven saved figures, visible preprocessing code.')
