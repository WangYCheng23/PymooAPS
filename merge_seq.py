# -*- encoding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

    
def Merge(abs_path, steps):    
    DATA_PATH = os.path.join(abs_path,'Data')
    paths = os.listdir(os.path.join(DATA_PATH, 'raw'))
    paths.sort( key=lambda x: '-'.join(x.split('/')[-1].split('.')[0].split('-')[-2:]))

    res = []
    for f_name in paths:  
        real_length = int(f_name.split('-')[0])  
        df_raw = pd.read_csv(os.path.join(DATA_PATH, 'raw',f_name))
        df_seq = pd.read_csv(os.path.join(DATA_PATH, f'output/{steps}',f'res_{f_name}'))
        assert df_raw.shape[0] == real_length, f"{f_name} raw data wrong, which is {df_raw.shape[0]}"
        
        X_np = df_seq.values[0].astype(int)
        if X_np.min() < 0 or X_np.max() >= len(df_raw):
            raise ValueError
        X_lt = X_np.tolist()
        assert len(X_lt)==real_length, f"{f_name} seq length wrong, which is {len(X_lt)}"
        df_res = df_raw.copy()
        df_res = df_res.iloc[X_lt]
        res.append(df_res)
        
    res_df = pd.concat(res, axis=0)
    res_df.to_csv(os.path.join(DATA_PATH,f'output/{steps}/result.csv'), index=False)    
    print("finished merge!")
 