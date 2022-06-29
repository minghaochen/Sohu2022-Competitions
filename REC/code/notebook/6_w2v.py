import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import gc
import re
import time
import matplotlib.pyplot as plt
import utils
%matplotlib inline
# import seaborn as sns
from pandarallel import pandarallel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec
pandarallel.initialize()


def train_w2v(prefix, embed_size, window_size, min_count, sentences):
    print('start....')
    start = time.time()
    w2v_model = Word2Vec(sentences=sentences, 
                     vector_size=embed_size, 
                     window=window_size, 
                     min_count=min_count, 
                     epochs=10,
                     sg=0, hs=1, seed=42, workers=8)
    end = time.time()
    print(f'end  use time {round((end - start) / (1000 * 60), 3)} min')
    
    path = f'../model/{prefix}_w2v_size{embed_size}_window{window_size}_mincount{min_count}.model'
    w2v_model.save(path)
    print(f'save to {path}')
    

data = pd.read_feather('../../data/Sohu2022_data/rec_data/dataB.feather')
userSeq = pd.read_csv('../../data/Sohu2022_data/rec_data/userSeqAll.csv')
userSeq['userSeq'] = '{' + userSeq['userSeq'].astype(str).values + '}'
userSeq['userSeq'] = userSeq['userSeq'].str.replace(";", ",")
userSeq['userSeq'] = userSeq['userSeq'].astype(str).parallel_apply(lambda x: eval(x) if x != '{nan}' else {})
data['itemId'] = data['itemId'].astype(str)

######### 不排序
userSeq['itemIdSeq'] = userSeq['userSeq'].parallel_apply(lambda x: list(x.keys()))
userSeq['itemIdSeq'] = userSeq['itemIdSeq'].parallel_apply(lambda x: [str(i) for i in x])
userSeq[['suv', 'itemIdSeq', 'userSeqFlag']].to_csv( f'../feats/user_itemIdSeq.csv', index=None)

sentences = userSeq['itemIdSeq'].tolist()
prefix = 'hist_click_seq'
train_w2v(prefix, embed_size=64, window_size=128, min_count=1, sentences=sentences)

itemIdSeq_agg = data.groupby('suv')['itemId'].agg(list).reset_index(name='itemIdSeq')
sentences = itemIdSeq_agg['itemIdSeq'].tolist()
prefix = 'rec_seq_groupby_suv'
train_w2v(prefix, embed_size=64, window_size=21, min_count=5, sentences=sentences)

itemIdSeq_agg_pvId = data.groupby('pvId')['itemId'].agg(list).reset_index(name='itemIdSeq_pvId')
sentences = itemIdSeq_agg_pvId['itemIdSeq_pvId'].tolist()
prefix = 'rec_seq_groupby_pvId'
train_w2v(prefix, embed_size=64, window_size=21, min_count=5, sentences=sentences)

itemIdSeq = pd.concat([userSeq[['itemIdSeq']], itemIdSeq_agg[['itemIdSeq']]])
itemIdSeq.to_csv('../tmp/w2v_itemIdSeq_train.csv', index=None)

##### 排序
value = userSeq['userSeq'].values.tolist()
value = list(map(lambda x: sorted(x.items(), key = lambda kv:kv[1], reverse=False), value))
userSeq['userSeq_sorted'] = value
userSeq['itemIdSeq_sorted'] = userSeq['userSeq_sorted'].parallel_apply(lambda x: [i[0] for i in x])
userSeq['itemIdSeq_sorted'] = userSeq['itemIdSeq_sorted'].parallel_apply(lambda x: [str(i) for i in x])
userSeq[['suv', 'itemIdSeq_sorted', 'userSeqFlag']].to_csv( f'../feats/user_itemIdSeq_sorted.csv', index=None)

sentences = userSeq['itemIdSeq_sorted'].tolist()
prefix = 'hist_click_seq_sorted'
train_w2v(prefix, embed_size=64, window_size=21, min_count=5, sentences=sentences)

values = data[['sampleId', 'pvId', 'logTs']].values.tolist()
values = sorted(values,key=(lambda x:[x[1],x[2]]),reverse=False)

new_df = pd.DataFrame(values, columns=['sampleId', 'pvId', 'logTs'])
del data['logTs']
new_df = new_df.merge(data, on=['sampleId', 'pvId'], how='left')

del data
data = new_df
gc.collect()

itemIdSeq_sorted_agg = data.groupby('suv')['itemId'].agg(list).reset_index(name='itemIdSeq_sorted')
sentences = itemIdSeq_sorted_agg['itemIdSeq_sorted'].tolist()
prefix = 'rec_seq_groupby_suv_sorted'
train_w2v(prefix, embed_size=64, window_size=21, min_count=5, sentences=sentences)

itemIdSeq_sorted = pd.concat([userSeq[['itemIdSeq_sorted']], 
                              itemIdSeq_sorted_agg[['itemIdSeq_sorted']]])
itemIdSeq_sorted.to_csv('../tmp/w2v_itemIdSeq_sorted_train.csv', index=None)
