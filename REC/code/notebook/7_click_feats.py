import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import re
import time
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from pandarallel import pandarallel
import utils
from sklearn.preprocessing import StandardScaler, LabelEncoder
pandarallel.initialize()


data = pd.read_feather('../../data/Sohu2022_data/rec_data/dataB.feather')
train = data[data.label.notnull()]
userSeq = pd.read_csv('../../data/Sohu2022_data/rec_data/userSeqAll.csv')

userSeq['userSeq'] = '{' + userSeq['userSeq'].astype(str).values + '}'
userSeq['userSeq'] = userSeq['userSeq'].str.replace(";", ",")
userSeq['userSeq'] = userSeq['userSeq'].astype(str).parallel_apply(lambda x: eval(x) if x != '{nan}' else {})

userSeq = userSeq.sort_values(['suv', 'logTs']).reset_index(drop=True)
userSeq = userSeq.drop_duplicates('suv', keep='last').reset_index(drop=True)

value = userSeq['userSeq'].values.tolist()
value = list(map(lambda x: sorted(x.items(), key = lambda kv:kv[1], reverse=False), value))
userSeq['userSeq_sorted'] = value
userSeq['itemIdSeq_sorted'] = userSeq['userSeq_sorted'].parallel_apply(lambda x: [i[0] for i in x])

def get_userSeq(row):
    userSeq = row.userSeq_sorted
    suv = row.suv
    res = []
    for i in userSeq:
        res.append([suv, i[0], i[1], 1])
    return res
userSeq['userSeq_tmp'] = userSeq[['suv', 'userSeq_sorted']].drop_duplicates('suv').parallel_apply(lambda x: get_userSeq(x), axis=1)

tmp = [i for s in userSeq['userSeq_tmp'].tolist() for i in s]
histclick = pd.DataFrame(data=tmp, columns=['suv', 'itemId', 'logTs', 'label'])
del tmp; gc.collect()

userBehavior = pd.concat([data[['suv', 'itemId', 'logTs', 'label']], histclick])

userBehavior.sort_values('logTs', inplace=True)
userBehavior.reset_index(drop=True, inplace=True)
userBehavior = userBehavior.fillna(-1)
userBehavior['label'] = userBehavior['label'].astype(np.int8)
userBehavior.to_feather('../tmp/userBehavior.feather')

all_item = userBehavior.copy()

userBehavior_click_dict = dict()
userBehavior_count_dict = dict()

ret = list()
for idx, row in tqdm(all_item.iterrows()):
    
    logTs = row.logTs
    itemId = row.itemId
    label = row.label        
    
    try:
        count = userBehavior_count_dict[itemId]
        click = userBehavior_click_dict[itemId]
        
        if (label != -1):   
            userBehavior_count_dict[itemId] += 1
        if (label == 1):
            userBehavior_click_dict[itemId] += 1
    except:
        count = 0
        click = 0
        
        if (label != -1):
            userBehavior_count_dict[itemId] = 1
        
        if (label == 1):
            userBehavior_click_dict[itemId] = 1
        if (label == 0):
            userBehavior_click_dict[itemId] = 0
    
    data = {
        'suv': row.suv,
        'itemId': row.itemId,
        'logTs': row.logTs,
        'label': row.label,
        'count': count,
        'click': click,
    }
    ret.append(data)
all_item = pd.DataFrame(ret)
all_item.rename(columns={'count':'hist_count', 'click': 'hist_click'}, inplace=True)
hist_count_click = all_item[['suv', 'itemId', 
                             'logTs', 'hist_count', 
                             'hist_click']].drop_duplicates(['suv', 'itemId', 'logTs'])

hist_count_click = hist_count_click.reset_index(drop=True)
hist_count_click.to_feather(f'../feats/itemId_hist_feats.feather')

histclick = histclick.merge(all_item[['suv', 'itemId', 'logTs', 'hist_count', 'hist_click']], 
                            on=['suv', 'itemId', 'logTs'],
                            how='left')

histclick_counts = histclick.groupby('suv')['hist_count'].agg(list).reset_index()
histclick_click = histclick.groupby('suv')['hist_click'].agg(list).reset_index()
histclick_itemId = histclick.groupby('suv')['itemId'].agg(list).reset_index()
histclick_logTs = histclick.groupby('suv')['logTs'].agg(list).reset_index()

userSeq_hist_feats = (histclick_counts
                 .merge(histclick_click, on='suv', how='left')
                 .merge(histclick_itemId, on='suv', how='left')
                 .merge(histclick_logTs, on='suv', how='left'))
feats = []
for col in ['count', 'click']:
    userSeq_hist_feats[f'hist_seq_{col}_max'] = userSeq_hist_feats[f'hist_{col}'].parallel_apply(lambda x: np.max(x))
    userSeq_hist_feats[f'hist_seq_{col}_min'] = userSeq_hist_feats[f'hist_{col}'].parallel_apply(lambda x: np.min(x))
    userSeq_hist_feats[f'hist_seq_{col}_std'] = userSeq_hist_feats[f'hist_{col}'].parallel_apply(lambda x: np.std(x))
    userSeq_hist_feats[f'hist_seq_{col}_mean'] = userSeq_hist_feats[f'hist_{col}'].parallel_apply(lambda x: np.mean(x))
    userSeq_hist_feats[f'hist_seq_{col}_median'] = userSeq_hist_feats[f'hist_{col}'].parallel_apply(lambda x: np.median(x))
    feats.extend([
        f'hist_seq_{col}_max', f'hist_seq_{col}_min', 
        f'hist_seq_{col}_std', f'hist_seq_{col}_mean', f'hist_seq_{col}_median'
    ])

    
userSeq_hist_feats = reduce_mem_usage(userSeq_hist_feats)
userSeq_hist_feats[['suv'] + feats].to_feather(f'../feats/hist_seq_feats.feather')

########### 用户点击序列 全局count click特征
userBehavior = pd.read_feather('../tmp/userBehavior.feather')
agg_rato = userBehavior.groupby('itemId')['label'].agg('mean').reset_index(name=f'overall_itemId_rato')
agg_click = userBehavior.groupby('itemId')['label'].agg('sum').reset_index(name=f'overall_itemId_click')
agg_count = userBehavior.groupby('itemId')['label'].agg('count').reset_index(name=f'overall_itemId_count')

overall_feats = agg_click.merge(agg_count, on='itemId', how='left')
overall_feats.to_feather(f'../feats/itemId_overall_feats.feather')

userSeq['itemIdSeq'] = userSeq['userSeq'].parallel_apply(lambda x: list(x.keys()))

count_dict = dict(zip(overall_feats['itemId'], overall_feats['overall_itemId_count']))
userSeq['overall_counts'] = userSeq['itemIdSeq'].parallel_apply(lambda x: [count_dict[i] for i in x])
click_dict = dict(zip(overall_feats['itemId'], overall_feats['overall_itemId_click']))
userSeq['overall_clicks'] = userSeq['itemIdSeq'].parallel_apply(lambda x: [click_dict[i] for i in x])


def get_feat(x, agg):
    if agg == 'max':
        return np.max(x)
    elif agg == 'min':
        return np.min(x)
    elif agg == 'median':
        return np.median(x)
    elif agg == 'std':
        return np.std(x)
    elif agg == 'mean':
        return np.mean(x)

feat_names = []
for col in ['count', 'click']:
    for agg in [
        'max', 'min', 'median',
        'std', 'mean', 
    ]:
        feat = f'overall_{col}_{agg}' 
        
        userSeq[feat] = (userSeq[f'overall_{col}s']
                         .parallel_apply(
                             lambda x: get_feat(x, agg) if len(x) != 0 else np.nan)
                        )
        
        userSeq[feat].fillna(userSeq[feat].mean(), inplace=True)
        feat_names.append(feat)
        
userSeq = reduce_mem_usage(userSeq)
userSeq[['suv'] + feat_names].to_feather(f'../feats/hist_seq_overall_feats.feather')
