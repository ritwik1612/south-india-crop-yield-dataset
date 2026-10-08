"""Validate exported training inputs and document measured preprocessing findings."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from denoising_autoencoder import DenoisingAutoencoder

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir',type=Path,default=ROOT)
    args=parser.parse_args(); root=args.project_dir
    master=pd.read_csv(root/'DATASET/merged/rice_master.csv')
    manifest=json.loads((root/'DATASET/training/manifest.json').read_text())
    assert master.shape==(2018,30)
    assert master[['IDREGION','FYEAR','SEASON']].duplicated().sum()==0
    report={'rows':len(master),'columns':len(master.columns),
            'state_counts':master['STATE'].value_counts().to_dict(),
            'season_counts':master['SEASON'].value_counts().to_dict(),
            'missing_master_cells':int(master.isna().sum().sum()),'representations':{}}
    for arm in ('standard','autoencoder'):
        info=manifest[arm]; predictors=info['predictor_columns']; split_info={}
        assert not any(name in predictors for name in ('YIELD','PRODUCTION_TONNES','IDREGION','DISTRICT','FYEAR'))
        for name,years in [('train',range(2001,2016)),('validation',range(2016,2018)),('test',range(2018,2020))]:
            data=pd.read_csv(root/'DATASET/training'/info[name])
            expected=master.loc[master['FYEAR'].isin(years)].reset_index(drop=True)
            assert len(data)==len(expected)
            assert data[['IDREGION','FYEAR','SEASON']].equals(expected[['IDREGION','FYEAR','SEASON']])
            assert np.allclose(data['YIELD'],expected['YIELD'])
            assert np.isfinite(data[predictors].to_numpy()).all()
            split_info[name]={'rows':len(data),'predictors':len(predictors)}
        report['representations'][arm]=split_info
    train=master.loc[master.FYEAR<=2015]
    checkpoint=torch.load(root/'pre-processing/artifacts/denoising_autoencoder.pt',map_location='cpu',weights_only=False)
    model=DenoisingAutoencoder(len(checkpoint['numeric_features']),checkpoint['latent_dim'],checkpoint['dropout'])
    model.load_state_dict(checkpoint['model_state_dict']); model.eval()
    values=checkpoint['numeric_preprocessor'].transform(train[checkpoint['numeric_features']])
    with torch.no_grad(): _,latent=model(torch.tensor(values[:5],dtype=torch.float32))
    exported=pd.read_csv(root/'DATASET/training/autoencoder/train.csv')
    names=[f'AE_LATENT_{i:02d}' for i in range(1,checkpoint['latent_dim']+1)]
    assert np.allclose(latent.numpy(),exported[names].head().to_numpy(),atol=1e-5)
    report['unseen_validation_test_seasons']=sorted(set(master.SEASON)-set(train.SEASON))
    report['checks_passed']=['split membership','label preservation','no duplicate master keys',
                              'finite predictors','explicit non-leaking predictor manifest',
                              'saved DAE checkpoint reproduces exported latent features']
    output=root/'docs/preprocessing/output_validation.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
