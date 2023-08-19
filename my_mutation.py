# -*- encoding: utf-8 -*-
import copy
import itertools
import random
import sys

import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.operators.crossover.ox import random_sequence


class single_day_Mutation(Mutation):
    def __init__(self, data_encode, opt1=False, prob=0.5):
        super().__init__()
        self.prob = prob
        self.data = data_encode
        self.date = data_encode['计划日期'].unique()[0]
        self.step = 0

    def _do(self, problem, X, **kwargs):
        # print('-------doing mutation-------')
        # X:[matings, DNA]
        Y = copy.deepcopy(X)
        date = self.date
        for i, y in enumerate(X):  
            prob = random.random()
            if prob <= 0.5:
                # 变异内部顺序
                # print('-------mutation1-------')
                # date = random.choice(self.dates)
                car_names = list(y[0].keys())
                if len(car_names)>=2:
                    random.shuffle(car_names[:])
                    tmp_dict = {key: y[0][key] for key in car_names}
                    # print(f'mutation shuffle car order in date ({date})')
                    y[0] = tmp_dict
                Y[i] = y
                
            elif prob >= 0.75:
                # print('-------mutation2-------')
                # 反向交换
                cars_name = list(y[0].keys()) # 当天车辆keys
                car_category = random.choice(cars_name)
                
                mu = y[0][car_category]
                n = len(mu)
                if n >= 2:
                    start,end = random_sequence(n)
                    # print(f'mutation reverse seq ({start},{end}) in car ({car_category}) at date ({date})')
                    mu[start:end + 1] = np.flip(mu[start:end + 1]).tolist()
                y[0][car_category] = mu  
                Y[i] = y
        self.step += 1
        return Y

