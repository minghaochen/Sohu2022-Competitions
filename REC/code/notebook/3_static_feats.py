import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import gc
import re
import time
import matplotlib.pyplot as plt
%matplotlib inline
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

def convert_count(x):
    if x>60000:
        return 0
    elif x>50000:
        return 1
    elif x>40000:
        return 2
    elif x>30000:
        return 3
    elif x>20000:
        return 4
    elif x>10000:
        return 5
    elif x>8000:
        return 6
    elif x>6000:
        return 7
    elif x>4000:
        return 8
    elif x>2000:
        return 9
    elif x>1000:
        return 10
    else:
        return 11
    
#### 当天热门文章 分类
itemId_by_day_count = (data
                       .groupby(['month_day', 'itemId'])['sampleId']
                       .agg('count')
                       .reset_index(name='itemId_by_day_count'))
itemId_by_day_count['itemId_by_day_count_class'] = (itemId_by_day_count['itemId_by_day_count']
                                                    .parallel_apply(lambda x: convert_count(x)))

data = data.merge(itemId_by_day_count, on=['month_day', 'itemId'], how='left')

data['itemId_suv_nunique'] = data.groupby(['itemId'])['suv'].transform('nunique')
data['suv_itemId_nunique'] = data.groupby(['suv'])['itemId'].transform('nunique')

data['itemId_pvId_nunique'] = data.groupby('itemId')['pvId'].transform('nunique')
data['pvId_itemId_nunique'] = data.groupby('pvId')['itemId'].transform('nunique')

data['pvId_suv_nunique'] = data.groupby('pvId')['suv'].transform('nunique')
data['suv_pvId_nunique'] = data.groupby('suv')['pvId'].transform('nunique')

def convert_itemId(x):
    if x>111000:
        return 0
    elif x>110000:
        return 1
    elif x>80000:
        return 2
    elif x>75000:
        return 3
    elif x>70000:
        return 4
    elif x>60000:
        return 5
    elif x>50000:
        return 6
    elif x>40000:
        return 7
    elif x>30000:
        return 8
    elif x>20000:
        return 9
    elif x>10000:
        return 10
    elif x>5000:
        return 11
    elif x>1000:
        return 12
    else:
        return 13
    
data['itemId_count'] = data.groupby(['itemId'])['sampleId'].transform('count')
data['itemId_class'] = data['itemId_count'].parallel_apply(lambda x: convert_itemId(x))
data['itemId_class'].unique()
data['logTs'] = data.groupby('pvId')['logTs'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
data['logTs'].fillna(0, inplace=True)

data['logTs_x'] = data['logTs'].parallel_apply(lambda x: 0 if x == 0 else 1)
data['pos_logTs'] = data['pos'].astype(str) + '_' + data['logTs_x'].astype(str)

data['pos_logTs_count'] = data.groupby('pos_logTs')['sampleId'].transform('count')

def convert_pos_logTs(x):
    if x.pos_logTs_count < 100:
        return '1'
    elif x.pos_logTs_count < 200:
        return '2'
    else:
        return x.pos_logTs
    
data['pos_logTs_class'] = data.parallel_apply(lambda x: convert_pos_logTs(x), axis=1)
data['pos_logTs_class'].unique()

data = data.merge(userSeq[['suv', 'userSeqFlag', 'userSeq']], 
                  on=['suv', 'userSeqFlag'], how='left'
                 )
data['new_itemId'] = data.parallel_apply(lambda x: 1 if str(x.itemId) in str(x.userSeq) else 0, axis=1)
del data['userSeq']; gc.collect()

def targetEncoder(features, data):
    
    feats = []
    train = data[data.label.notnull()]
    test = data[data.label.isnull()]
    train['label'] = train['label'].astype(np.int8)
    
    for f in features:
        agg_rato = train.groupby(f)['label'].agg('mean').reset_index()
        agg_rato.rename(columns={'label': f'{f}_rato'}, inplace=True)
        
        agg_click = train.groupby(f)['label'].agg('sum').reset_index()
        agg_click.rename(columns={'label': f'{f}_click'}, inplace=True)
        
        agg_count = train.groupby(f)['sampleId'].agg('nunique').reset_index()
        agg_count.rename(columns={'sampleId': f'{f}_count'}, inplace=True)
        
        ######## train 
        train = train.merge(agg_rato, on=f, how='left')
        train[f'{f}_rato'].fillna(agg_rato[f'{f}_rato'].mean(), inplace=True)
        
        train = train.merge(agg_click, on=f, how='left')
        train[f'{f}_click'].fillna(agg_click[f'{f}_click'].mean(), inplace=True)
        
        train = train.merge(agg_count, on=f, how='left')
        train[f'{f}_count'].fillna(agg_count[f'{f}_count'].mean(), inplace=True)

        ####### test    
        test = test.merge(agg_rato, on=f, how='left')
        test[f'{f}_rato'].fillna(agg_rato[f'{f}_rato'].mean(), inplace=True)
        
        test = test.merge(agg_click, on=f, how='left')
        test[f'{f}_click'].fillna(agg_click[f'{f}_click'].mean(), inplace=True)
        
        test = test.merge(agg_count, on=f, how='left')
        test[f'{f}_count'].fillna(agg_count[f'{f}_count'].mean(), inplace=True)
        
        feats.extend([f'{f}_rato', f'{f}_click', f'{f}_count'])
    data = pd.concat([train, test])
    return feats, data

targetFeats = [
    'suv', 'pvId',
    
    'operator', 'browserType', 'deviceType', 'osType', 
    'province', 'city', 
    
    'itemId_by_day_count_class',
    'new_itemId',
    'itemId_class', 
    'pos_logTs_class',
    'pos',
]

feats, data = targetEncoder(targetFeats, data)

feats = ['suv_rato','suv_click','pvId_rato','pvId_click','operator_rato','operator_click',
 'browserType_rato','browserType_click','deviceType_rato','deviceType_click','osType_rato','osType_click',
 'province_rato','province_click','city_rato','city_click','new_itemId_rato','new_itemId_click',
 'itemId_class_rato','itemId_class_click','pos_logTs_class_rato','pos_logTs_class_click','pos_rato','pos_click']

usefeats = [

'sampleId',
    
'itemId_class', 'new_itemId', 
            
'itemId_count', 'suv_count', 'pvId_count',

'operator_count', 'browserType_count', 'deviceType_count',
'osType_count', 'province_count', 'city_count',

'itemId_suv_nunique', 'suv_itemId_nunique', 
'itemId_pvId_nunique','pvId_itemId_nunique', 
'pvId_suv_nunique', 'suv_pvId_nunique',
    
'itemId_by_day_count_class_count',
          
       ]

data.reset_index(drop=True, inplace=True)

data = reduce_mem_usage(data)

data[usefeats + feats].to_feather('../feats/static_feats.feather')