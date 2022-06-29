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
import utils

data = pd.read_feather('../../data/Sohu2022_data/rec_data/dataB.feather')
userSeq = pd.read_csv('../../data/Sohu2022_data/rec_data/userSeqAll.csv')

data['pvId_count'] = data.groupby('pvId')['pvId'].transform('count')
count = data[['pvId', 'pvId_count']].drop_duplicates('pvId')
pos_values = []
for i in count['pvId_count'].values:
    pos_values.extend(list(range(i)))
    
data['pos'] = pos_values
del data['pvId_count']

values = data[['sampleId', 'pvId', 'logTs']].values.tolist()
values = sorted(values,key=(lambda x:[x[1],x[2]]),reverse=False)

new_df = pd.DataFrame(values, columns=['sampleId', 'pvId', 'logTs'])
del data['logTs']
new_df = new_df.merge(data, on=['sampleId', 'pvId'], how='left')

del data
gc.collect()

data = new_df
gc.collect()

group = data.groupby('pvId')['logTs']
# 距离上一次最近的一一次正样本的gap
data['gap'] = group.shift(0) - group.shift(1)
del group

# 这个不知道该不该填均值
data['gap'] = data['gap'].fillna(data['gap'].mean())

data['gap'] = data.groupby('pvId')['gap'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
data['gap'].fillna(0, inplace=True)

# 构造序列特征 前后窗口7
# gap pos itemId
# itemId 的滑窗不一定有用
l = 7
timing_cols = []
cate_feats = ['pos', 'itemId']
for i in range(l * 2 + 1):
    data['gap_%s' % (i - l)] = data.groupby('pvId')['gap'].shift(i - l)
    data['gap_%s' % (i - l)] = data['gap_%s' % (i - l)].fillna(0)
    timing_cols += ['gap_%s' % (i - l)]
    
#     data['logTs_%s' % (i - l)] = data.groupby('pvId')['logTs'].shift(i - l)
#     data['logTs_%s' % (i - l)] = data['logTs_%s' % (i - l)].fillna(0)
#     timing_cols += ['logTs_%s' % (i - l)]

    for cate in cate_feats:
        new_col = f'{cate}_{(i - l)}'
        if cate in ['pos', 'itemId']:
            data[new_col] = data.groupby('pvId')[cate].shift(i - l).fillna(-1)
        else:
            data[new_col] = data[cate]
        data[new_col] = data[new_col].astype(int)
        timing_cols += [new_col]
        
        
data = reduce_mem_usage(data)

data = data.sort_values('sampleId')
data.reset_index(drop=True,inplace=True)

data[['sampleId', 'pos', 'gap'] + timing_cols].to_feather('../feats/pos_gap_feats_all_gapmixmin.feather')
