# -*- encoding: utf-8 -*-
import itertools
import random
import sys
import warnings
import joblib

import numpy as np
import pandas as pd
from tqdm import tqdm
import math

from itertools import groupby
import datetime
import os
from copy import deepcopy
import time
from multiprocessing import Pool


from utils import df_encode, group_elements, split_list
from obj_func import ObjFunc


warnings.filterwarnings('ignore')

def preprocess(data):
    group_list = []
    df = data.copy()

    for _, order_g in df.groupby(["计划日期", "车型", "天窗", "外色描述", "四驱车", "K3", "小颜色", "大颜色", "双颜色", "石墨电池", "车辆等级描述", "电池特征", "order"], as_index=False):
        subset_index = order_g['sort_index'].tolist()
        sorted(subset_index)
        group_list.append(subset_index)
    random.shuffle(group_list)
    return group_list

def sub_fixed_seq(data):
    """固定前后车型"""
    data_encode = df_encode(data, downsample=True)
    date_n = data_encode['计划日期'].unique()[0]

    date_list = dict()
    # 预处理
    for car_n, car_g in data_encode.groupby("车型", as_index=False):
        car_order = ['K'+str(car_id) for car_id in data_encode['车型'].unique().tolist()]        
        # 预分组
        car_list = preprocess(car_g)
        date_list['K'+str(car_n)] = car_list 
    random.shuffle(car_order)         
    seq_list = {k:date_list[k] for k in car_order}

    return seq_list

def random_seq(data):
    """随机初始化"""
    data_encode = df_encode(data, downsample=True)
    
    groups = data_encode.groupby("车型")
    group_list = [g for _,g in groups]
    np.random.shuffle(group_list)
    seq = []
    for g in group_list:
        shuffle_g = g.sample(frac=1).reset_index(drop=True)
        seq_ind = shuffle_g["sort_index"].to_list()
        seq += seq_ind
    return seq

def init_x(data, pop_size=8, seed=1, n_jobs=1):
    """初始策略"""
    X = []
    
    seqs = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(sub_fixed_seq)(data) for _ in range(pop_size))
    print('finish initilize')
    
    X = np.array([[seq] for seq in seqs])

    return X

    