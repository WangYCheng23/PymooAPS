# -*- encoding: utf-8 -*-
import os
import sys

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV


def best_divisor(dividend, divisor_range, mod_range):
    div, mod = divmod(dividend, divisor_range)
    is_available = np.where((mod == 0) | ((mod >= mod_range[0]) & (mod <= mod_range[1])), 1, 0)
    div_ceil = div + np.where(mod > 0, 1, 0)
    out = sorted(zip(is_available, div_ceil, mod, divisor_range), key=lambda x: [-x[0], x[1], x[2], -x[3]])
    return out[0][-1]

div_dict = {"小颜色": [30] * 2000, "大颜色": [30] * 2000, "双颜色": [4] * 2000}
for i in range(0, 2000):
    div_dict['小颜色'][i] = best_divisor(i, np.arange(30, 14, -1), [15, 30])
    div_dict['大颜色'][i] = best_divisor(i, np.arange(30, 14, -1), [15, np.inf])
    div_dict['双颜色'][i] = best_divisor(i, np.arange(4, 0, -1), [0, 4])
 
def df_encode(data, downsample=False):
    df_code_dict = {
        '车型':{'K1':1, 'K2':2, 'K3':3, 'K4':4, 'K5':5, 'K6':6, 'K7':7, 'K8':8},
        '天窗':{'天幕':1, '整体式全景天窗':2, '无天窗':3, '全景EC天幕':4},
        '外色描述':{
            # --------------------------------小颜色 ----------------------------------
            '冰玫粉':1, '冰玫粉-Y':1, '松花黄':2, '松花黄-Y':2, '幻光紫':4, '幻光紫-Y':4, \
            '星漫绿':5, '星漫绿-Y':5, '天际灰':6, '量子红':7, '烟雨青':8, '脉冲蓝':9, 
            # --------------------------------大颜色 ----------------------------------    
            '自由灰':-1, '自由灰-Y':-1, '极地白':-2, '极地白-Y':-2, '素雅灰':-3, '素雅灰-Y':-3, '夜影黑':-4, '夜影黑-Y':-4,\
            '天青色':-5, '天青色-Y':-5, '极速银':-6, '极速银-Y':-6, '极速银(出租车)':-7, '全息银':-8, '白云蓝':-9,
            # --------------------------------双颜色 --------------------------------    
            '黑/全息银':100, '黑/极地白':200, '黑/幻光紫':300, '黑/星漫绿':400, '黑/烟雨青':500, '黑/冰玫粉':600, '黑/极地白-Y':700, '黑/天青色-Y':800, '黑/量子红':900   
            },
        '车辆等级描述':{'高配':1, '次中配':2, '次高配':3, '高配科技':4, '次高配科技':5, '中配':6, '低配-Lite':7, '低配':8, '网约车北方版':9, '出租车全国版':10, \
                        '顶配':11, '出租车标准版北方版':12, '标配七座版':13, '中配-Lite':14, '出租车定制版北方版':15, '低配七座版':16},
        '电池特征':{
                    '厂商A+厂商B 93.3kWh+180/180kW':-1,   #四驱                 
                    '厂商G(石墨烯)+厂商B 70.4kWh+165kW':0,     #石墨电池               
                    '厂商A+厂商B 80kWh+165kW':1,
                    '厂商A(磷酸铁锂)+厂商B 71.8kWh+165kW':2,
                    '厂商D(磷酸铁锂）174Ah（500）+厂商E135kW':3, 
                    '厂商A 218Ah（610）+厂商B76.8kWh+135kW':4,
                    '厂商C+厂商B 69.9kWh+165kW（新）':5,
                    '厂商A LFP 177Ah（460）+厂商B(小) 60.038kWh+100kW(厂商FBMS)':6,
                    '厂商A+厂商B 93.3kWh+180kW':7, 
                    '厂商A 58Ah(510)+厂商B(小)58.81kWh+150kW':8,
                    '厂商A LFP 132Ah（410 HCU-4）+厂商B(小) 50.688kWh+100kW':9,
                    '厂商A58(580)+厂商B（小） 58.81kWh+100kW':10, 
                    '厂商A 58Ah(510)+厂商B 58.81kWh+150kW':11,
                    '厂商C 58.5Ah(510)+厂商B（小）58.931kWh+150kW':12,
                    '厂商D LFP 174Ah（500）+厂商B64.6kWh+135kW':13, 
                    '厂商A58(530)+厂商B（小） 58.81kWh+100kW':14,
                    '厂商C 58.5Ah(602)+厂商B69.94kWh+165kW':15, 
                    '厂商D LFP 174Ah（500）+厂商B61.7kWh+150kW':16,
                    '厂商D LFP 174Ah（500）+厂商E61.7kWh＋150kW':17,
                    '厂商A 58Ah（460）+厂商B(小) 58.81kWh+100kW':18, 
                    '厂商A 218Ah（610）+厂商B72kWh+150kW':19,
                    '厂商A LFP 177Ah（510）+厂商E（小）60.038KWh+150kW(厂商FBMS)':20,
                    '厂商D(磷酸铁锂）174Ah（510）+厂商E(模块化）150kW':21
                }
    }
    df = data.copy()
    df= df[["生产订单号-ERP", "计划日期", "每日顺序", "车型", "天窗", "外色描述", "车辆等级描述", "电池特征"]]
    df["sort_index"] = df.reset_index().index.values
    for col in df_code_dict.keys():
        df[col] = df[col].apply(lambda x: df_code_dict[col][x])
        
    df["四驱车"] = df["电池特征"].apply(lambda x: 1 if x==-1 else 0)
    df["K3"] = df["车型"].apply(lambda x: 1 if x==3 else 0)
    df["小颜色"] = df["外色描述"].apply(lambda x: 1 if 0<x<10 else 0)
    df["大颜色"] = df["外色描述"].apply(lambda x: 1 if x<0 else 0)
    df["双颜色"] = df["外色描述"].apply(lambda x: 1 if x>=100 else 0)
    df["石墨电池"] = df["电池特征"].apply(lambda x: 1 if x==0 else 0)   
     
    # downsample <- baseline  
    if downsample: 
        attribute = ["计划日期", "车型", "天窗", "外色描述", "四驱车", "K3", "小颜色", "大颜色", "双颜色", "石墨电池", "电池特征", "车辆等级描述"]
        df = df.sort_values(by=["计划日期", "车型", "天窗", "四驱车", "K3", "外色描述", '车辆等级描述', '电池特征', "每日顺序"]).reset_index(drop=True)
        df["num"] = df.groupby(attribute)["每日顺序"].transform("count").values
        df["rank"] = df.groupby(attribute)["每日顺序"].transform("rank").values
        df['order'] = np.ceil(df['rank'] / 30)
        for col in ['大颜色', '小颜色', '双颜色']:
            df['order'] = np.where(df[col] == 1, np.ceil(df['rank']/df['num'].apply(lambda x: div_dict[col][int(x)])), df['order'])
        df["order"] = np.where(df["石墨电池"]==1, df["rank"], df["order"])
        df['num'] = 1
        df = df.drop(["rank"], axis=1)
    return df
 
    
def csv_find(parent_path, file_flag='.csv', all_dirs=False):
    '''批处理读csv文件位置'''
    df_paths = []
    if all_dirs:
        for root, dirs, files in os.walk(parent_path):  # os.walk输出[目录路径，子文件夹，子文件]
            for file in files:
                if "".join(file).find(file_flag) != -1:  # 判断是否csv文件(files是list格式，需要转成str)
                    df_paths.append(root + '/' + file)  # 所有csv文件
    else:
        for file in os.listdir(parent_path):
            if "".join(file).find(file_flag) != -1:
                df_paths.append(parent_path + '/' + file)
    return df_paths

def output_csv(steps, abs_path, paths):
    res = []
    for path in paths:
        if np.random.random()<0.9:
            df = pd.read_csv(os.path.join(abs_path,'Data/interim/split_a',path), index_col=0)
        else:
            df = pd.read_csv(os.path.join(abs_path,'Data/interim/split_b',path), index_col=0)
        res.append(df)
    res_df = pd.concat(res)
    res_df.to_csv(os.path.join(abs_path,f'Data/output/{steps}/result.csv'), index=False)

def split_list(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def count_batch_nums(element_array):
    list_batch_nums = []
    for x in element_array:
        if x[0] != 0:
            list_batch_nums.append(len(x))
    return list_batch_nums

def count_interval_satisfication(element_split, interval_nums):
    # list_interval_nums = []
    group = 0
    satisfaction = 0
    left = 0
    while left<len(element_split):
        if element_split[left][0] == 0:
            break
        left += 1
        group += 1
    right = left+2
    while right<len(element_split):
        if element_split[left][0]==0 and element_split[right][0]==0:
            if not left==0 and not right==len(element_split)-1:
                if len(element_split[left])>=interval_nums and len(element_split[right])>=interval_nums:
                    satisfaction += 1
            elif left==0 and len(element_split[right])>=interval_nums:
                satisfaction += 1
            elif len(element_split[left])>=interval_nums and right==len(element_split)-1:
                satisfaction += 1
            # list_interval_nums.append(len(element_split[left+1]))
        left += 1
        if element_split[left][0]!=0:
            group += 1 
        right += 1
    return satisfaction/group if group!=0 else 1.

def group_elements(element_array):
    """切分相同元素"""
    element_split = []
    # element_split_id = [0]
    element_temp = [element_array[0]]
    for i in range(1,len(element_array)):
        if element_array[i] == element_array[i-1]:
            element_temp.append(element_array[i])
        else:
            element_split.append(element_temp)
            element_temp = [element_array[i]]
            # element_split_id.append(i)
    element_split.append(element_temp)
    # element_split_id = [element_split_id[i]-element_split_id[i-1] for i in range(1, len(element_split_id))]
    return element_split
