# -*- encoding: utf-8 -*-
import itertools
from typing import List, Dict
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from obj_func import ObjFunc
import math


# Problem   
class SortOpt(ElementwiseProblem):
    def __init__(self, data, print_flag=0, **kwargs):
        """
        n_var :  Number of Variables
        n_obj :  Number of Objectives
        n_ieq_constr : Number of Inequality Constraints
        n_eq_constr :  Number of Equality Constraints
        xl :  Lower bounds for the variables. if integer all lower bounds are equal.
        xu :  Upper bounds for the variable. if integer all upper bounds are equal.
        vtype : The variable type. So far, just used as a type hint.
        """
        n_var = 1
        n_obj = 5
        self.df = data
        self.date = data['计划日期'].unique()[0]
        self.df_len = len(data)
        self.print_flag = print_flag
        self.obj = ObjFunc(print_flag)
        # 人工baseline
        self.f_baseline = self.obj.cal_baselines(self.df)
        # self.baseline_score = sum(self.f_baseline)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0, xu=n_var, vtype=int, **kwargs)
        
        
    def _evaluate(self, x, out, *args, **kwargs):
        """
        x: 输入排序
        out: 输出结果得分
        --------------------
        "F": function values
        "G": constraints
        """
        # 展开  
        x = x[0]  
        seq = []
        for car_k, car_v in x.items():
                seq += list(itertools.chain(*car_v))       
        X = self.df.copy().iloc[seq]
        f0, f1, f2, f3 = self.obj.cal_obj(X)
        # sort_score = sum[f0, f1, f2, f3, f4]
        if self.print_flag == 1:
            print('原始响应:', f0, f1, f2, f3)
        
        res_f0, res_f1, res_f2, res_f3 = [],[],[],[]
        
        #--------------------------------------------------------------------------------------#
        for obj in ["car_category_switch", "window_switch", "color_switch", "battery_switch", "car_level_descriptiion_switch"]:
            if self.f_baseline[0][obj] == 0:
                s = (self.f_baseline[0][obj] - f0[obj])
            else:    
                s = (self.f_baseline[0][obj] - f0[obj])/(self.f_baseline[0][obj])
            res_f0.append(s)
        #--------------------------------------------------------------------------------------# 
        for obj in ["four_wheel_centralization", "K3_centralization"]:
            if self.f_baseline[1][obj] == 0:
                s = (self.f_baseline[1][obj] - f1[obj])
            else:    
                s = (self.f_baseline[1][obj] - f1[obj])/(self.f_baseline[1][obj])
            res_f1.append(s)
        #--------------------------------------------------------------------------------------#
        for obj in ["small_color_interval_satisfaction", "dual_color_interval_satisfaction", "graphite_interval_satisfaction"]:
            if self.f_baseline[2][obj] == 0:
                s = (math.exp(math.exp(f2[obj])))
            else:
                s = (f2[obj] - self.f_baseline[2][obj])/(self.f_baseline[2][obj])
            res_f2.append(s)
        #--------------------------------------------------------------------------------------#        
        for obj in ["small_color_batch_satisfaction", "dual_color_batch_satisfaction", "big_color_batch_satisfaction", "graphite_batch_satisfaction"]:
            if self.f_baseline[3][obj] == 0:
                s = (math.exp(math.exp(f3[obj])))
            else:
                s = (f3[obj] - self.f_baseline[3][obj])/(self.f_baseline[3][obj])
            res_f3.append(s)
        #--------------------------------------------------------------------------------------#
        
        # #单目标计算
        T = res_f2+res_f3
        cmp1 = res_f0[0]
        cmp2 = np.min(T)
        cmp3 = (4*res_f0[1]+2*res_f0[2]+sum(res_f1)+sum(res_f2)+sum(res_f3))
        cmp4 = np.minimum(np.mean(T),0.8)
        cmp5 = sum(res_f0[-2:])
        out["F"] = [cmp5,cmp4,cmp3,cmp2,cmp1]
        
        return out
    
