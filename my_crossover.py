# -*- encoding: utf-8 -*-
import random
import copy
import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.operators.crossover.ox import ox, random_sequence


def my_ox(receiver, donor, seq=None, shift=False):
    assert len(donor) == len(receiver)
    
    seq = seq if not None else random_sequence(len(receiver))
    start, end = seq
    
    # the final value to be returned
    y = []

    # the donation and a set of it to allow a quick lookup
    donation = np.copy(donor[start:end + 1]).tolist()
    
    for k in range(len(receiver)):

        # do the shift starting from the swapped sequence - as proposed in the paper
        i = k if not shift else (start + k) % len(receiver)
        v = receiver[i]

        flag = False
        for d in donation:
            if sorted(v) == sorted(d):
                flag = True
        if flag == False:   
            y.append(v)
            
    y = y[:start]+donation+y[start:]
    return y

class single_day_Crossover(Crossover):
    def __init__(self, data_encode, shift=False, prob=1, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.shift = shift
        self.prob = prob
        self.data = data_encode
        self.date = data_encode['计划日期'].unique()[0]
        self.step = 0
        
    def _do(self, problem, X, **kwargs):
        # print('-------doing crossover-------')
        _, n_matings, n_var = X.shape
        # Y = np.full((self.n_offsprings, n_matings, n_var), -1, dtype=int)
        Y = copy.deepcopy(X)
        prob = np.random.random()
        date = self.date
        # print(f'----------------------{date}:{self.step}----------------------')
        
        for i in range(n_matings):
            if prob < self.prob:
                ###################只交叉一天###################            
                # 顺序选取
                cars_name = list(Y[0, i][0].keys()) # 当天车辆keys
                car_category = random.choice(cars_name)

                a = Y[0, i][0][car_category]
                b = Y[1, i][0][car_category]
                
                assert len(a)==len(b), 'parents length unequal!'
                n = len(a)
                if n >= 2:
                    start, end = random_sequence(n)
                    # print(f'crossover car category {car_category} from ({start}-{end}) at date ({date})')
                    y_a = my_ox(a, b, seq=(start, end), shift=self.shift)
                    y_b = my_ox(b, a, seq=(start, end), shift=self.shift) 
                    
                    assert len(y_a)==len(y_b), 'childrens length unequal!'
                    
                    Y[0, i][0][car_category] = y_a
                    Y[1, i][0][car_category] = y_b
                    
        self.step += 1
        return Y

