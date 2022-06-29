import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import gc
import re
import time
import matplotlib.pyplot as plt
%matplotlib inline
# import seaborn as sns
from pandarallel import pandarallel
from sklearn.preprocessing import StandardScaler, LabelEncoder
pandarallel.initialize()

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

data = pd.read_feather('../../data/Sohu2022_data/rec_data/dataB.feather')
userSeq = pd.read_csv('../../data/Sohu2022_data/rec_data/userSeqAll.csv')

userSeq['userSeq'] = '{' + userSeq['userSeq'].astype(str).values + '}'
userSeq['userSeq'] = userSeq['userSeq'].str.replace(";", ",")
userSeq['userSeq'] = userSeq['userSeq'].astype(str).parallel_apply(lambda x: eval(x) if x != '{nan}' else {})
value = userSeq['userSeq'].values.tolist()
value = list(map(lambda x: sorted(x.items(), key = lambda kv:kv[1], reverse=False), value))
tsSeq_value = list(map(lambda seq: [i[1] for i in seq], value))
userSeq['tsSeq_sorted'] = tsSeq_value
userSeq['tsSeq_sorted'] = userSeq['tsSeq_sorted'].parallel_apply(lambda x:[float(i) for i in x])

def gapSeq(x):
    if len(x) <= 1:
        return []
    res = []
    n = len(x)
    for i in range(n - 1):
        res.append(x[i + 1] - x[i])
    return res

userSeq['gapSeq'] = userSeq['tsSeq_sorted'].parallel_apply(lambda x: gapSeq(x))

userSeq['gapSeq_mean']  = userSeq['gapSeq'].parallel_apply(lambda x: np.mean(x))
userSeq['gapSeq_std'] = userSeq['gapSeq'].parallel_apply(lambda x: np.std(x))
userSeq['time_last'] = userSeq['tsSeq_sorted'].parallel_apply(lambda x: x[-1] if len(x)>0 else np.nan)
userSeq['time_first'] = userSeq['tsSeq_sorted'].parallel_apply(lambda x: x[0] if len(x)>0 else np.nan)
userSeq['time_last'] = userSeq['time_last'].fillna(userSeq['time_last'].mean())
userSeq['time_first'] = userSeq['time_first'].fillna(userSeq['time_first'].mean())

userSeq['time_long'] = userSeq['time_last'] - userSeq['time_first']
userSeq['time_long'] = userSeq['time_long'].astype('float32')

#####
userSeq['month_day'] = userSeq['tsSeq_sorted'].parallel_apply(lambda x:[time.strftime("%m-%d", time.localtime(i/1000 + 28800)) for i in x])
# 计算不同天
userSeq['month_day_nunique'] = userSeq['month_day'].apply(lambda x: len(set(x)) if len(x)>0 else 0).astype('int8')
del userSeq['month_day'];gc.collect()
userSeq['hour'] = userSeq['tsSeq_sorted'].parallel_apply(lambda x:[time.strftime("%H", time.localtime(i/1000 + 28800)) for i in x])
userSeq['hour'] = userSeq['hour'].parallel_apply(lambda x: [int(i) for i in x])
# 计算不同小时
userSeq['hour_nunique'] = userSeq['hour'].apply(lambda x: len(set(x)) if len(x)>0 else 0).astype('int8')
del userSeq['hour'];gc.collect()

# 计算两次关键词时间差均值 最大值 最小值
def diff(x):
    if len(x) <= 1:
        return 0
    diff_list = []
    for i in range(len(x) - 1):
        diff_list.append(x[i + 1] - x[i])
    return diff_list

userSeq['time_diff_mean'] = userSeq['tsSeq_sorted'].parallel_apply(lambda x: np.mean(diff(x))).astype('float32')
userSeq['time_diff_max'] = userSeq['tsSeq_sorted'].parallel_apply(lambda x: np.max(diff(x))).astype('float32')
userSeq['time_diff_min'] = userSeq['tsSeq_sorted'].parallel_apply(lambda x: np.min(diff(x))).astype('float32')
userSeq['time_diff_var'] = userSeq['tsSeq_sorted'].parallel_apply(lambda x: np.var(diff(x))).astype('float32')

# 计算时间的峰度和偏度
from scipy import stats

userSeq['time_skew'] = userSeq['tsSeq_sorted'].parallel_apply(lambda x: stats.skew(x) if len(x)>1 else 0).astype('float32')
userSeq['time_kurtosis'] = userSeq['tsSeq_sorted'].parallel_apply(lambda x: stats.kurtosis(x) if len(x)>1 else 0).astype('float32')

userSeq['seqLength'] = userSeq['userSeq'].parallel_apply(len)

feats = ['suv', 'userSeqFlag', 'tsSeq_sorted',
         'time_last', 'time_first', 'time_long', 
         'month_day_nunique', 'hour_nunique', 
         'time_diff_mean', 'time_diff_max', 'time_diff_min', 'time_diff_var',
         'time_skew', 'time_kurtosis',
         'gapSeq', 'gapSeq_mean', 'gapSeq_std', 'seqLength',
        ]

data = data.merge(userSeq[feats], 
                  on=['suv', 'userSeqFlag'], how='left'
                 )

data['time_gap_last'] = data['logTs'] - data['time_last']
data['time_gap_last'] = data['time_gap_last'].astype('float32')

data['time_gap_first'] = data['logTs'] - data['time_first']
data['time_gap_first'] = data['time_gap_first'].astype('float32')

data.reset_index(drop=True,inplace=True)

def get(x):
    if x.seqLength == 0:
        return x.gapSeq
    else:
        x
data['gapSeq'] = data.parallel_apply(lambda x: get(x), axis=1)

feats_save = ['sampleId',
         'time_last', 'time_first', 'time_long', 
         'month_day_nunique', 'hour_nunique', 
         'time_diff_mean', 'time_diff_max', 'time_diff_min', 'time_diff_var',
         'time_skew', 'time_kurtosis', 
         'time_gap_last', 'time_gap_first',
         'gapSeq', 'gapSeq_mean', 'gapSeq_std', 'seqLength',
        ]
data[feats_save].to_feather('../feats/time_feats.feather')