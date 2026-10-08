"""One command: three raw sources -> one master -> standard and DAE training splits."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir',type=Path,default=ROOT)
    parser.add_argument('--epochs',type=int,default=180)
    args=parser.parse_args(); project=args.project_dir.resolve(); scripts=Path(__file__).resolve().parent
    dataset=project/'DATASET'; artifacts=scripts/'artifacts'; figures=project/'docs/figures'; reports=project/'docs/preprocessing'
    for path in (artifacts,figures,reports): path.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(scripts/'build_dataset.py'),'--dataset-dir',str(dataset)],check=True)
    if project == ROOT:
        subprocess.run([sys.executable,str(scripts/'make_figures.py')],check=True)
    subprocess.run([sys.executable,str(scripts/'preprocess_master.py'),'--master-path',str(dataset/'merged/rice_master.csv'),
                    '--output-dir',str(dataset/'training/standard'),'--artifacts-dir',str(artifacts)],check=True)
    deep=dataset/'training/autoencoder'; deep.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(scripts/'denoising_autoencoder.py'),'--model-ready-path',str(dataset/'merged/rice_master.csv'),
                    '--output-dir',str(deep),'--epochs',str(args.epochs)],check=True)
    for split in ('train','validation','test'):
        (deep/f'rice_yield_{split}_deep_processed.csv').replace(deep/f'{split}.csv')
    for path in list(deep.iterdir()):
        if path.suffix=='.pt': destination=artifacts/'denoising_autoencoder.pt'
        elif path.suffix=='.png': destination=figures/path.name
        elif path.name not in ('train.csv','validation.csv','test.csv'): destination=reports/path.name
        else: continue
        shutil.move(str(path),str(destination))
    report=json.loads((reports/'deep_preprocessing_report.json').read_text(encoding='utf-8'))
    categories=json.loads((artifacts/'standard_manifest.json').read_text(encoding='utf-8'))['predictor_columns']
    predictor_columns=[f'AE_LATENT_{i:02d}' for i in range(1,report['latent_dim']+1)]
    # OneHotEncoder names in the DAE transformer match the standard transformer.
    predictor_columns += [name for name in categories if name.startswith('categorical__')]
    predictor_columns += ['AE_RECONSTRUCTION_MSE']
    manifest={'standard':{'train':'standard/train.csv','validation':'standard/validation.csv','test':'standard/test.csv',
                          'predictor_columns':categories},
              'autoencoder':{'train':'autoencoder/train.csv','validation':'autoencoder/validation.csv','test':'autoencoder/test.csv',
                             'predictor_columns':predictor_columns},
              'target':'YIELD','metadata_columns':['IDREGION','STATE','DISTRICT','FYEAR','SEASON'],
              'split_rows':report['split_rows']}
    (dataset/'training/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Ready: DATASET/training/standard and DATASET/training/autoencoder. See manifest.json for safe predictor selection.')

if __name__=='__main__': main()
