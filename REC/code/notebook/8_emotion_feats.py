import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import random
import tensorflow as tf
import gc
import re
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
%matplotlib inline
import seaborn as sns
from pandarallel import pandarallel
import utils
from sklearn import preprocessing
from sklearn.cluster import KMeans,MiniBatchKMeans
from tensorflow.keras.preprocessing import text, sequence

from sklearn.preprocessing import StandardScaler, LabelEncoder
pandarallel.initialize()
tqdm.pandas(desc='apply')

# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
seed_everything(2022)

rec_A = pd.read_csv('../../data/Sohu2022_data/rec_data/recommend_content_entity_0317.txt', sep='\t', header=None)
rec_sup = pd.read_csv('../../data/Sohu2022_data/rec_data/recommend_content_entity_sup.txt', sep='\t', header=None)
rec_B = pd.read_csv('../../data/Sohu2022_data/rec_data/recommend_content_entity_B.txt', sep='\t', header=None)
rec_entity = pd.concat([rec_A, rec_sup, rec_B])

rec_entity[0] =  rec_entity[0].map(eval)
features = ['id', 'entity']
for feat in features:
    rec_entity[feat] = rec_entity[0].map(lambda x: x[feat])
del rec_entity[0];gc.collect()
rec_entity_dict = dict(zip(rec_entity['id'], rec_entity['entity']))
del rec_entity; gc.collect()

rec_entity_emotion_1 = pd.read_csv('../feats/rec_entity_train.txt', sep='\t')
rec_entity_emotion_2 = pd.read_csv('../feats/rec_entity_test.txt', sep='\t')
rec_entity_emotion_3 = pd.read_csv('../feats/rec_entity.csv')
rec_entity_emotion = pd.concat([rec_entity_emotion_1, rec_entity_emotion_2, rec_entity_emotion_3])
rec_entity_emotion['result'] = rec_entity_emotion['result'].map(eval)

def get_emotion(x):
    emotion_dict = dict()
    for key, value in x.items():
        emotion_dict[key] = np.argmax(value) - 2
    return emotion_dict
rec_entity_emotion['emotion'] = rec_entity_emotion['result'].parallel_apply(lambda x: get_emotion(x))

rec_entity_emotion_dict = dict(zip(rec_entity_emotion['id'].astype(str), rec_entity_emotion['emotion']))

data = pd.read_feather('../../data/Sohu2022_data/rec_data/dataB.feather')
userSeq = pd.read_csv('../../data/Sohu2022_data/rec_data/userSeqAll.csv')

userSeq['userSeq'] = '{' + userSeq['userSeq'].astype(str).values + '}'
userSeq['userSeq'] = userSeq['userSeq'].str.replace(";", ",")
userSeq['userSeq'] = userSeq['userSeq'].astype(str).parallel_apply(lambda x: eval(x) if x != '{nan}' else {})

value = userSeq['userSeq'].values.tolist()
value = list(map(lambda x: sorted(x.items(), key = lambda kv:kv[1], reverse=False), value))
userSeq['userSeq_sorted'] = value
userSeq['itemIdSeq_sorted'] = userSeq['userSeq_sorted'].parallel_apply(lambda x: [i[0] for i in x])

hot_itemId = dict()
itemId_by_day_count = (data
                       .groupby(['month_day', 'itemId'])['sampleId']
                       .agg('count')
                       .reset_index(name='itemId_by_day_count'))

itemId_by_day_count = itemId_by_day_count.sort_values(['month_day', 'itemId_by_day_count'], ascending=False)

for month_day in itemId_by_day_count.month_day.unique():
    hot_itemId[month_day] = itemId_by_day_count[itemId_by_day_count.month_day == month_day]['itemId'][:10].tolist()
    
hot_itemId_entity = dict()
for key, value in hot_itemId.items():
    entitys = []
    for itemId in value:
        try:
            entitys.extend(rec_entity_dict[str(itemId)])
        except:
            pass
    hot_itemId_entity[key] = entitys
    
def get_entity(x):
    res = []
    if len(x) == 0:
        return []
    else:
        for itemId in x:
            try:
                itemId_entitys = rec_entity_dict[str(itemId)]
                res.extend(itemId_entitys)
            except:
                pass
    return res

userSeq['hist_entity'] = userSeq['itemIdSeq_sorted'].progress_apply(lambda x: get_entity(x))

def get_recent_entity(x):
    res = []
    if len(x) == 0:
        return []
    else:
        if len(x) < 3:
            window = 0
        else:
            window = 3
        for itemId in x[-window: ]:
            try:
                itemId_entitys = rec_entity_dict[str(itemId)]
                res.extend(itemId_entitys)
            except:
                pass
    return res

userSeq['recent3_entity'] = userSeq['itemIdSeq_sorted'].progress_apply(lambda x: get_recent_entity(x))

## 用户自身的偏好标签（取5个）
def user_preference_hist_tag(entitys, num=5):
    entity_count = dict()
    for entity in entitys:
        try:
            entity_count[entity] += 1
        except:
            entity_count[entity] = 1
    ## 排序
    entity_sorted = sorted(entity_count.items(),key = lambda x:x[1],reverse = False)
    
    res = [i[0] for i in entity_sorted[: min(num, len(entity_sorted))] ]
    return res
userSeq['hist_entity_most5'] = userSeq['hist_entity'].progress_apply(lambda x: user_preference_hist_tag(x))
userSeq['recent3_entity_most5'] = userSeq['recent3_entity'].progress_apply(lambda x: user_preference_hist_tag(x))

def get_rec_entity(x):
    try:
        itemId_entitys = rec_entity_dict[str(x)]
        return itemId_entitys
    except:
        return []
data['rec_entity'] = data['itemId'].progress_apply(lambda x: get_rec_entity(x))

data = data.merge(userSeq[['suv', 'userSeqFlag', 'itemIdSeq_sorted', 
                           'recent3_entity', 'hist_entity', 'hist_entity_most5', 'recent3_entity_most5']],
                  on=['suv', 'userSeqFlag'], how='left')
del userSeq; gc.collect()

for f in ['operator', 'browserType', 
     'deviceType', 'osType', 
     'province', 'city']:
    del data[f]
gc.collect()

## 情感极性
# 推荐文章的实体，是否在历史点击实体中出现过，出现过计算情感的相似程度，

def have_hist_entity_emotion(x, window=True):
    
    rec_entity = x.rec_entity
    itemId = str(x.itemId)
    histIds = x.itemIdSeq_sorted
    
    try:
        _ = rec_entity_emotion_dict[itemId]
    except:
        return np.nan
    
    
    if len(rec_entity) == 0:
        return np.nan
    
    if window==None:
        window=0
        
    emotion_similary = 0
    n = 0
    for entity in rec_entity:
        emotion = rec_entity_emotion_dict[itemId][entity]
        for histId in histIds[-window:]:
            try:
                hist_emotion = rec_entity_emotion_dict[str(histId)][entity]
                emotion_similary += (hist_emotion - emotion) ** 2
                n += 1
            except:
                continue
    if n == 0:
        return 0
    return emotion_similary / n

data['hist_emotion_sim'] = data.progress_apply(lambda x: have_hist_entity_emotion(x, None), axis=1)
data['hist_emotion_sim'].fillna(data['hist_emotion_sim'].astype(np.float32).mean(), inplace=True)
data['recent3_emotion_sim'] = data.progress_apply(lambda x: have_hist_entity_emotion(x, 3), axis=1)
data['recent3_emotion_sim'].fillna(data['recent3_emotion_sim'].astype(np.float32).mean(), inplace=True)

### 是否在当天热门文章（10篇）中出现过，计算情感的相似程度
### 是否在前天热门文章（10篇）中出现过，计算情感的相似程度

def have_hot_entity_emotion(x, mode='today'):
    
    itemId = str(x.itemId)
    rec_entity = x.rec_entity
    
    day = int(x.month_day[-1])
    today = x.month_day
    yesterday = f'01-0{day - 1}'
    
    
    if mode == 'today':
        histIds = hot_itemId[today]
    else:
        if yesterday == '01-00':
            return np.nan
        histIds = hot_itemId[yesterday]
    
    try:
        _ = rec_entity_emotion_dict[itemId]
    except:
        return np.nan
    
    
    if len(rec_entity) == 0:
        return np.nan
    
    emotion_similary = 0
    n = 0
    
    for entity in rec_entity:
        emotion = rec_entity_emotion_dict[itemId][entity]
        for histId in histIds:
            try:
                hist_emotion = rec_entity_emotion_dict[str(histId)][entity]
                emotion_similary += (hist_emotion - emotion) ** 2
                n += 1
            except:
                continue
    
    if n == 0:
        return 0
    return emotion_similary / n

data['today_hot_emotion_sim'] = data.progress_apply(lambda x: have_hot_entity_emotion(x, 'today'), axis=1)
data['today_hot_emotion_sim'].fillna(data['today_hot_emotion_sim'].astype(np.float32).mean(), inplace=True)

data['yesterday_hot_emotion_sim'] = data.progress_apply(lambda x: have_hot_entity_emotion(x, 'yesterday'), axis=1)
data['yesterday_hot_emotion_sim'].fillna(data['yesterday_hot_emotion_sim'].astype(np.float32).mean(), inplace=True)

feats = ['sampleId', 'hist_emotion_sim', 'recent3_emotion_sim', 'today_hot_emotion_sim',
       'yesterday_hot_emotion_sim']
data[['sampleId'] + feats].to_feather('../feats/emotions_feats.feather')