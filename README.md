# The 2nd Guangzhou • Pazhou Algorithm Competition -- Intelligent production scheduling algorithm based on multi-objective and multi-source data disturbance prediction

# Table of Contents

- [The 2nd Guangzhou • Pazhou Algorithm Competition -- Intelligent production scheduling algorithm based on multi-objective and multi-source data disturbance prediction](#the-2nd-guangzhou--pazhou-algorithm-competition----intelligent-production-scheduling-algorithm-based-on-multi-objective-and-multi-source-data-disturbance-prediction)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Running Time](#running-time)
- [Details](#details)

# Installation

```
conda create -n <env-name> python==3.8.15
pip install -r requirements.txt
```

# Usage

```
cd 工厂智能排产算法赛题-第3名-不得命咯

# 配置pymoo_main.py 第38-41行 参数
# 根据硬件配置调整 BATCH 和 N_PROCESS  
# 调整迭代次数 STEPS

bash run.sh

# check the results in ./submission/final_res.csv
```

# Running Time
```
# all depends on STEPS you config and Hardware you have
about 9 hours    # default config
```

# Details
```
cat details.txt
```
