# -*- encoding: utf-8 -*-
import pandas
import os
import sys
import pandas as pd


def Transform(abs_path, res_loc):
    df_res = pd.read_csv(os.path.join(abs_path, res_loc), index_col=0)
    res_seq = df_res['生产订单号-ERP'].apply(lambda x: int(x[3:])-1).to_list()

    df_raw = pd.read_csv(os.path.join(abs_path,'aps_data_B.csv'), index_col=0)
    df_raw['序号'] = df_raw.index
    df_raw = df_raw.sort_values('生产订单号-ERP').reset_index(drop=True)
    df_raw = df_raw.iloc[res_seq]
    df_raw = df_raw.set_index('序号', drop=True)
    df_raw.to_csv(os.path.join(abs_path,'submission/final_res.csv'))
    
