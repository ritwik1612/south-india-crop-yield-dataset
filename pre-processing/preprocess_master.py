"""Produce conventional training CSVs from the same 21 predictors used by the DAE."""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from denoising_autoencoder import select_numeric_features, chronological_split, one_hot_encoder, CATEGORICAL_FEATURES, METADATA_COLUMNS

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master-path',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--artifacts-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True); args.artifacts_dir.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(args.master_path).dropna(subset=['YIELD'])
    numeric=select_numeric_features(data); splits=chronological_split(data)
    processor=ColumnTransformer([
        ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),numeric),
        ('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',one_hot_encoder())]),CATEGORICAL_FEATURES),
    ])
    processor.fit(splits['train'][numeric+CATEGORICAL_FEATURES])
    names=processor.get_feature_names_out().tolist()
    for name,split in splits.items():
        features=pd.DataFrame(processor.transform(split[numeric+CATEGORICAL_FEATURES]),columns=names)
        result=pd.concat([split[METADATA_COLUMNS].reset_index(drop=True),features],axis=1)
        result.to_csv(args.output_dir/f'{name}.csv',index=False)
    with (args.artifacts_dir/'standard_preprocessor.pkl').open('wb') as f: pickle.dump(processor,f)
    report=dict(source=str(args.master_path),numeric_features=numeric,categorical_features=CATEGORICAL_FEATURES,
                predictor_columns=names,target='YIELD',split_rows={k:len(v) for k,v in splits.items()})
    (args.artifacts_dir/'standard_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
