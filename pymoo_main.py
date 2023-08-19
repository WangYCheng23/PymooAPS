# -*- encoding: utf-8 -*-
import itertools
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from pymoo.config import Config
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from tqdm import tqdm

from obj_strategy import init_x
from pymoo_algorithm import algorithm_choose
from utils import csv_find, output_csv
from pymoo_problem import SortOpt

pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 180)

import dill
import multiprocessing
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
from pymoo.core.callback import Callback

from merge_seq import Merge
from transform import Transform


POP_SIZE = 64
BATCH = 18
N_PROCESS = 4
STEPS = 10000     
  
        
def opt_data(data, name, baseline_path=None, verbose=False):
    start = time.time()
    print(f'-----{name}:优化开始-----')
    
    # initialize the thread pool and create the runner
    n_proccess = N_PROCESS
    pool = multiprocessing.Pool(n_proccess) 
    runner = StarmapParallelization(pool.starmap)
    
    # 自定义初始种群
    pop_size = POP_SIZE
    n_gen    = STEPS
    
    # 确定业务问题
    problem = SortOpt(data, elementwise_runner=runner)
    
    if baseline_path:
        df_res = pd.read_csv(baseline_path)
        X = df_res.index.values
    else:
        X = init_x(data, pop_size=pop_size)
    
    algorithm = algorithm_choose(X, data, case="GA", pop_size=pop_size)
    termination = ("n_gen", n_gen)
    
    # minimize优化逼近
    res = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=termination,
        seed=None,
        verbose=verbose,
        output=None
    )

    print(f'-----{name}:优化完成-----')
    F = res.pop.get("F")
    S = np.lexsort(F.transpose())[::-1]
    x = res.pop[S[0]].x[0]
    seq = []
    for _, car_v in x.items():
        seq += list(itertools.chain(*car_v))
    df_output = pd.DataFrame(np.array(seq).reshape(1,-1), columns=[f'Variable {i}' for i in range(1, len(seq)+1)])   
    
    end = time.time()
    code_time = int(end - start)
    print(f'{name} opt cost time: {(end - start):.0f}s')
    time.sleep(0.2)
    return df_output, code_time
    
def opt_main(opt_path, cwd_path, baseline=1, baseline_path=None, verbose=False):
    name = opt_path.split(r'/')[-1].split('.')[0]
    data = pd.read_csv(opt_path, index_col=0)
        
    # main function
    df_output, code_time = opt_data(data, name, baseline_path=baseline_path, verbose=verbose)

    # if verbose is False:
    abs_path = opt_path.split('/raw')[0]
    if not os.path.exists(f'{abs_path}/output/{STEPS}'):
        os.mkdir(f'{abs_path}/output/{STEPS}')
    df_output.to_csv(f'{abs_path}/output/{STEPS}/res_{name}.csv', index=False)
    return df_output, name, code_time

def applyParallel(cwd_path, paths, func):
    ret = joblib.Parallel(n_jobs=BATCH)(joblib.delayed(func)(opt_path, cwd_path) for opt_path in paths)
    return ret


if __name__ == '__main__':
    cwd_path = __file__ 
    if sys.platform.startswith('win32'):
        cwd_path = cwd_path[:cwd_path.rfind('\\')]
    else:
        cwd_path = cwd_path[:cwd_path.rfind('/')]
    
    start_all = time.time()
    abs_path = os.path.join(cwd_path,'Data')
    paths = csv_find(abs_path + '/raw')
    paths.sort(reverse=True, key=lambda x: int(x.split('/')[-1].split('.')[0].split('-')[0]))
    print("paths length:{}".format(len(paths)))

    for ii in range(len(paths)//BATCH+1):
        output = applyParallel(cwd_path, paths[ii*BATCH:(ii+1)*BATCH], opt_main)
    
    Merge(cwd_path, STEPS)
    Transform(cwd_path, f"Data/output/{STEPS}/result.csv")
    
    print('opt finish!!!')
    print(f'总时长: {(time.time() - start_all):.0f}s')