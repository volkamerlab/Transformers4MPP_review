import os.path

import pandas as pd


def calculate_pct_positive_class(data_path, output_name):
    for df_name in output_name:
        if df_name == 'tox21':
            df = pd.read_csv(os.path.join(data_path, f'{df_name}.csv'), index_col=-1).drop('mol_id', axis=1)
        else:
            df = pd.read_csv(os.path.join(data_path, f'{df_name}.csv'), index_col=0)
        pos_pct = []
        num_mols = []
        for col in df.columns:
            se = df[col].dropna()
            pos = len(se[se == 1])
            neg = len(se[se == 0])
            pos_pct.append(round(((pos / (pos + neg))*100), 0))
            num_mols.append(pos + neg)
        (pd.DataFrame({'endpoint': df.columns,
                      'pos_pct': pos_pct,
                      'num_mols': num_mols})
         .sort_values('pos_pct')
         .to_csv(os.path.join(data_path, f'{df_name}_pos_pct.csv'), index=False))

