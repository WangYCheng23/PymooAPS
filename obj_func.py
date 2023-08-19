# -*- encoding: utf-8 -*-
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd
import tqdm

warnings.filterwarnings('ignore')

from utils import df_encode, group_elements, count_interval_satisfication, count_batch_nums, csv_find


class ObjFunc():
    def __init__(self, print_flag=0):
        self.print_flag = print_flag
        
    def switch(self, data):
        """切换计算f"""
        data["车型切换"] = data["车型"].diff().apply(lambda x: 1 if x!=0 else 0)
        data["天窗切换"] = data["天窗"].diff().apply(lambda x: 1 if x!=0 else 0)
        data["外色描述切换"] = data["外色描述"].diff().apply(lambda x: 1 if x!=0 else 0)
        data["电池特征切换"] = data["电池特征"].diff().apply(lambda x: 1 if x!=0 else 0)
        data["车辆等级描述切换"] = data["车辆等级描述"].diff().apply(lambda x: 1 if x!=0 else 0)
        
        car_category_switch = data["车型切换"].sum()-1
        window_switch = data["天窗切换"].sum()-1
        color_switch = data["外色描述切换"].sum()-1
        battery_switch = data["电池特征切换"].sum()-1
        car_level_descriptiion_switch = data["车辆等级描述切换"].sum()-1
        
        if self.print_flag:
            print("="*20+"切换次数"+"="*20)
            print("车型切换次数: {}".format(car_category_switch))
            print("天窗切换次数: {}".format(window_switch))
            print("外色描述切换次数: {}".format(color_switch))
            print("电池特征切换次数: {}".format(battery_switch))
            print("车辆等级描述切换次数: {}".format(car_level_descriptiion_switch))
        
        f = {"car_category_switch":car_category_switch, "window_switch":window_switch, "color_switch":color_switch, "battery_switch":battery_switch, "car_level_descriptiion_switch":car_level_descriptiion_switch}
        return f
    
    def full_switch(self, data):
        res = []
        count = 0
        for n, g in data.groupby('计划日期',as_index=False):
            tmp = self.switch(g)
            res.append(list(tmp.values()))
            count += 1
        f_res = np.sum(res, axis=0)
        
        car_category_switch = f_res[0]
        window_switch = f_res[1]
        color_switch = f_res[2]
        battery_switch = f_res[3]
        car_level_descriptiion_switch = f_res[4]
        
        if self.print_flag:
            print("="*20+"切换次数"+"="*20)
            print("车型切换次数: {}".format(car_category_switch))
            print("天窗切换次数: {}".format(window_switch))
            print("外色描述切换次数: {}".format(color_switch))
            print("电池特征切换次数: {}".format(battery_switch))
            print("车辆等级描述切换次数: {}".format(car_level_descriptiion_switch))
        
        f = {"car_category_switch":car_category_switch, "window_switch":window_switch, "color_switch":color_switch, "battery_switch":battery_switch, "car_level_descriptiion_switch":car_level_descriptiion_switch}
        return f
    
    def centralization(self, data):
        """集中度切换计算f"""
        data["四驱集中"] = data["电池特征"].apply(lambda x: 1 if x==-1 else 0).diff().apply(lambda x: 1 if x!=0 else 0)
        data["K3集中"] = data["车型"].apply(lambda x: 1 if x==3 else 0).diff().apply(lambda x: 1 if x!=0 else 0)
        
        four_wheel_centralization = (data["四驱集中"].sum()-1)//2
        K3_centralization = (data["K3集中"].sum()-1)//2
        
        if self.print_flag:
            print("="*20+"集中度切换次数"+"="*20)
            print("四驱集中度: {}".format(four_wheel_centralization))
            print("K3集中度: {}".format(K3_centralization))
            
        f = {"four_wheel_centralization":four_wheel_centralization, "K3_centralization":K3_centralization}
        return f
            
    def interval(self, data):
        """间隔计算满足f"""
        data["小颜色隔"] = data["外色描述"].apply(lambda x: x if 0<x<10 else 0)
        data["双颜色隔"] = data["外色描述"].apply(lambda x: x if x>=100 else 0)
        data["石墨电池隔"] = data["电池特征"].apply(lambda x: 1 if x==0 else 0)
        
        small_color_split = group_elements(data["小颜色隔"].to_numpy())
        dual_color_split = group_elements(data["双颜色隔"].to_numpy())
        graphite_split = group_elements(data["石墨电池隔"].to_numpy())
        
        small_color_interval_satisfaction = count_interval_satisfication(small_color_split, 60)
        dual_color_interval_satisfaction = count_interval_satisfication(dual_color_split, 60)
        graphite_interval_satisfaction = count_interval_satisfication(graphite_split, 30)
        
        # small_color_interval_satisfaction = round(small_color_interval_satisfaction,3)
        # dual_color_interval_satisfaction = round(dual_color_interval_satisfaction,3)
        # graphite_interval_satisfaction = round(graphite_interval_satisfaction,3)
        
        # if self.print_flag:
        #     print("="*20+"间隔满足率"+"="*20)
        #     print("小颜色间隔: {}".format(small_color_interval_satisfaction))
        #     print("双颜色间隔: {}".format(dual_color_interval_satisfaction))
        #     print("石墨电池间隔: {}".format(graphite_interval_satisfaction))
        
        f = {"small_color_interval_satisfaction":small_color_interval_satisfaction, "dual_color_interval_satisfaction":dual_color_interval_satisfaction, "graphite_interval_satisfaction":graphite_interval_satisfaction}
        return f

    def full_interval(self, data):
        res = []
        count = 0
        for n, g in data.groupby('计划日期',as_index=False):
            tmp = self.interval(g)
            res.append(list(tmp.values()))
            count += 1
        f_res = np.sum(res, axis=0)/count
        
        small_color_interval_satisfaction = round(f_res[0],3)
        dual_color_interval_satisfaction = round(f_res[1],3)
        graphite_interval_satisfaction = round(f_res[2],3)
        
        if self.print_flag:
            print("="*20+"间隔满足率"+"="*20)
            print("小颜色间隔: {}".format(small_color_interval_satisfaction))
            print("双颜色间隔: {}".format(dual_color_interval_satisfaction))
            print("石墨电池间隔: {}".format(graphite_interval_satisfaction))
        
        f = {"small_color_interval_satisfaction":small_color_interval_satisfaction, "dual_color_interval_satisfaction":dual_color_interval_satisfaction, "graphite_interval_satisfaction":graphite_interval_satisfaction}
        return f
    
    def batch(self, data):
        """批数计算满足f"""
        data["小颜色批"] = data["外色描述"].apply(lambda x: x if 0<x<10 else 0)
        data["双颜色批"] = data["外色描述"].apply(lambda x: x if x>=100 else 0)
        data["石墨电池批"] = data["电池特征"].apply(lambda x: 1 if x==0 else 0)
        
        small_color_split = group_elements(data["小颜色批"].to_numpy())
        dual_color_split = group_elements(data["双颜色批"].to_numpy())
        graphite_split = group_elements(data["石墨电池批"].to_numpy())
        
        data["大颜色批"] = data["外色描述"].apply(lambda x: x if x<0 else 0)
        big_color_split = group_elements(data["大颜色批"].to_numpy())
        # big_color_interval_nums = count_interval_nums(big_color_split)
        
        small_color_batch_nums = count_batch_nums(small_color_split)
        dual_color_batch_nums = count_batch_nums(dual_color_split)
        big_color_batch_nums = count_batch_nums(big_color_split)
        graphite_batch_nums = count_batch_nums(graphite_split)
        
        small_color_batch_satisfaction = sum([1 if 15<=batch<=30 else 0 for batch in small_color_batch_nums])/len(small_color_batch_nums) if len(small_color_batch_nums)>0 else 1.
        dual_color_batch_satisfaction = sum([1 if batch<=4 else 0 for batch in dual_color_batch_nums])/len(dual_color_batch_nums) if len(dual_color_batch_nums)>0 else 1.
        big_color_batch_satisfaction = sum([1 if 15<=batch else 0 for batch in big_color_batch_nums])/len(big_color_batch_nums) if len(big_color_batch_nums)>0 else 1.
        graphite_batch_satisfaction = sum([1 if batch<=1 else 0 for batch in graphite_batch_nums])/len(graphite_batch_nums) if len(graphite_batch_nums)>0 else 1.
        
        small_color_batch_satisfaction = round(small_color_batch_satisfaction,3)
        dual_color_batch_satisfaction = round(dual_color_batch_satisfaction,3)
        big_color_batch_satisfaction = round(big_color_batch_satisfaction,3)
        graphite_batch_satisfaction = round(graphite_batch_satisfaction,3)
        
        if self.print_flag:
            print("="*20+"批数满足率"+"="*20)
            print("小颜色批数: {}".format(small_color_batch_satisfaction))
            print("双颜色批数: {}".format(dual_color_batch_satisfaction))
            print("大颜色批数: {}".format(big_color_batch_satisfaction))
            print("石墨电池批数: {}".format(graphite_batch_satisfaction))
        
        f = {"small_color_batch_satisfaction":small_color_batch_satisfaction, "dual_color_batch_satisfaction":dual_color_batch_satisfaction, "big_color_batch_satisfaction":big_color_batch_satisfaction, "graphite_batch_satisfaction":graphite_batch_satisfaction}
        return f
    
    def uniformity(self, data):
        """均匀性计算"""
        # data["双颜色"] = data["外色描述"].apply(lambda x: x if x>=100 else 0)
        # data["石墨电池"] = data["电池特征"].apply(lambda x: 1 if x==0 else 0)
        data["K3_uniform"] = data["车型"].apply(lambda x: 1 if x==3 else 0).diff().apply(lambda x: 1 if x!=0 else 0)
        
        dual_color_split = group_elements(data["双颜色"].to_numpy())
        graphite_split = group_elements(data["石墨电池"].to_numpy())
        K3_split = group_elements(data["K3_uniform"].to_numpy())
        
        dual_color_uniformity = np.std(count_batch_nums(dual_color_split))/np.mean(count_batch_nums(dual_color_split))
        graphite_uniformity = np.std(count_batch_nums(graphite_split))/np.mean(count_batch_nums(graphite_split))
        K3_uniformity = np.std(count_batch_nums(K3_split))/np.mean(count_batch_nums(K3_split))
        
        if np.isnan(dual_color_uniformity): dual_color_uniformity=0.
        if np.isnan(graphite_uniformity): graphite_uniformity=0.
        if np.isnan(K3_uniformity): K3_uniformity=0.
        
        if self.print_flag:
            print("="*20+"均匀性计算"+"="*20)
            print("双颜色均匀性:{}".format(dual_color_uniformity))
            print("石墨电池均匀性:{}".format(graphite_uniformity))
            print("K3均匀性:{}".format(K3_uniformity))
        
        f = {"dual_color_uniformity":dual_color_uniformity, "graphite_uniformity":graphite_uniformity, "K3_uniformity":K3_uniformity}
        return f
            
    def cal_baselines(self, df):
        """人工 baselines"""
        data_encode = df_encode(df)
        data_baseline = data_encode.sort_values(by=['生产订单号-ERP'])
        f0 = self.switch(data_baseline)
        f1 = self.centralization(data_baseline)
        f2 = self.interval(data_baseline)
        f3 = self.batch(data_baseline)
        return [f0, f1, f2, f3]
        
    def cal_obj(self, df):
        """种群结果"""
        data_encode = df_encode(df)
        f0 = self.switch(data_encode)
        f1 = self.centralization(data_encode)
        f2 = self.interval(data_encode)
        f3 = self.batch(data_encode)
        return [f0, f1, f2, f3]
    
    def cal_score(self, df):
        data_encode = df_encode(df)
        f0 = self.switch(data_encode)
        f1 = self.full_interval(data_encode)
        f2 = self.batch(data_encode)
        return [f0, f1, f2,]
    
    def obj_series(self, df):
        d = dict()
        obj = self.cal_obj(df)
        for f in obj:
            for k,v in f.items():
                d[k] = v
        s = pd.Series(d)
        return s
    
