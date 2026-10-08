"""Create EDA figures from the single current master dataset."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parents[1]

def main():
    data=pd.read_csv(ROOT/'DATASET/merged/rice_master.csv'); dest=ROOT/'docs/figures'
    dest.mkdir(parents=True,exist_ok=True); sns.set_theme(style='whitegrid')
    fig,ax=plt.subplots(figsize=(10,5))
    sns.countplot(data=data,x='STATE',hue='SEASON',ax=ax)
    ax.set(title='Rice observations by state and season',ylabel='Observations',xlabel='')
    ax.tick_params(axis='x',rotation=12);fig.tight_layout();fig.savefig(dest/'01_observation_coverage.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));sns.boxplot(data=data,x='STATE',y='YIELD',ax=ax,color='#95B58D')
    ax.set(title='Rice-yield distribution',ylabel='Yield (tonnes/hectare)',xlabel='')
    ax.tick_params(axis='x',rotation=12);fig.tight_layout();fig.savefig(dest/'02_yield_distribution.png',dpi=170);plt.close(fig)
    annual=data.groupby(['FYEAR','STATE'],as_index=False)['YIELD'].mean()
    fig,ax=plt.subplots(figsize=(10,5));sns.lineplot(data=annual,x='FYEAR',y='YIELD',hue='STATE',marker='o',ax=ax)
    ax.set(title='Mean recorded rice yield over time',ylabel='Yield (tonnes/hectare)',xlabel='Crop year')
    fig.tight_layout();fig.savefig(dest/'03_annual_yield_by_state.png',dpi=170);plt.close(fig)
    columns=['YIELD','TAVG_MEAN','PREC_TOTAL','RAD_TOTAL','RELH_MEAN','GDD_BASE10','CROP_AREA']
    fig,ax=plt.subplots(figsize=(8,6));sns.heatmap(data[columns].corr(),cmap='BrBG',center=0,annot=True,fmt='.2f',ax=ax)
    ax.set_title('Feature correlations (association, not causation)')
    fig.tight_layout();fig.savefig(dest/'04_feature_correlation.png',dpi=170);plt.close(fig)
    print('Created four master-dataset EDA figures in docs/figures/.')

if __name__=='__main__': main()
