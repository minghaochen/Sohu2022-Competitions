import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import gc
import re
import time
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from pandarallel import pandarallel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.decomposition import TruncatedSVD
pandarallel.initialize()
import utils

data = pd.read_feather('../../data/Sohu2022_data/rec_data/dataB.feather')
userSeq = pd.read_csv('../../data/Sohu2022_data/rec_data/userSeqAll.csv')

userSeq['userSeq'] = '{' + userSeq['userSeq'].astype(str).values + '}'
userSeq['userSeq'] = userSeq['userSeq'].str.replace(";", ",")
userSeq['userSeq'] = userSeq['userSeq'].astype(str).parallel_apply(lambda x: eval(x) if x != '{nan}' else {})

###### hist_itemId_Seq不排序
userSeq['itemIdSeq'] = userSeq['userSeq'].parallel_apply(lambda x: list(x.keys()))
userSeq = userSeq.sort_values(['suv', 'logTs']).reset_index(drop=True)
tfidf_df = userSeq[['suv', 'itemIdSeq']].drop_duplicates('suv', keep='last').reset_index(drop=True)
tfidf_df['itemIdSeq'] = tfidf_df['itemIdSeq'].apply(lambda x:[str(i) for i in x])
tfidf_df['itemIdSeq_temp'] = tfidf_df['itemIdSeq'].apply(lambda x:' '.join(x))

print('=========Tfidf========')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.decomposition import TruncatedSVD

tfidf   = TfidfVectorizer(max_df=0.95, min_df=5, ngram_range=(1, 3))
res     = tfidf.fit_transform(tfidf_df['itemIdSeq_temp']) 

n_components = 32
print('svd start')
svd     = TruncatedSVD(n_components=n_components, n_iter=10, random_state=2022)
svd_res = svd.fit_transform(res)
print('svd finished')

tfidf_feats = []
for i in (range(n_components)):
    tfidf_df['tfidf_svd_{}'.format(i)] = svd_res[:, i]
    tfidf_df['tfidf_svd_{}'.format(i)] = tfidf_df['tfidf_svd_{}'.format(i)].astype('float32')
    tfidf_feats.append('tfidf_svd_{}'.format(i))

del tfidf_df['itemIdSeq_temp'],tfidf,res,svd,svd_res
gc.collect()

hist_itemId_Seq_tfidf = tfidf_df[['suv'] + tfidf_feats].sort_values('suv')[tfidf_feats].values
path = f'../feats/hist_click_seq_tfidf_size{n_components}_groupby_suv.npy'
np.save(path, hist_itemId_Seq_tfidf)

##### groupby pvId: itemId tfidf
pvId_tfidf_df = data.groupby('pvId')['itemId'].agg(list)
pvId_tfidf_df = pvId_tfidf_df.reset_index()

%%time
tfidf_df = pvId_tfidf_df[['pvId', 'itemId']].drop_duplicates('pvId').reset_index(drop=True)
tfidf_df['itemId'] = tfidf_df['itemId'].apply(lambda x:[str(i) for i in x])

tfidf_df['itemId_temp'] = tfidf_df['itemId'].apply(lambda x:' '.join(x))

print('=========Tfidf========')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.decomposition import TruncatedSVD

tfidf   = TfidfVectorizer(max_df=0.95, min_df=5, ngram_range=(1, 3))
res     = tfidf.fit_transform(tfidf_df['itemId_temp']) 

n_components = 32
print('svd start')
svd     = TruncatedSVD(n_components=n_components, n_iter=10, random_state=2022)
svd_res = svd.fit_transform(res)
print('svd finished')

tfidf_feats = []
for i in (range(n_components)):
    tfidf_df['groupby_pvId_tfidf_svd_{}'.format(i)] = svd_res[:, i]
    tfidf_df['groupby_pvId_tfidf_svd_{}'.format(i)] = tfidf_df['groupby_pvId_tfidf_svd_{}'.format(i)].astype('float32')
    tfidf_feats.append('groupby_pvId_tfidf_svd_{}'.format(i))
del tfidf_df['itemId_temp'],tfidf,res,svd,svd_res
gc.collect()

rec_itemId_tfidf = tfidf_df[['pvId'] + tfidf_feats].sort_values('pvId')[tfidf_feats].values
path = f'../feats/rec_seq_tfidf_size{n_components}_groupby_pvId.npy'
np.save(path, rec_itemId_tfidf)

#####  groupby suv: itemId tfidf
suv_tfidf_df = data.groupby('suv')['itemId'].agg(list)
suv_tfidf_df = suv_tfidf_df.reset_index()
%%time
tfidf_df = suv_tfidf_df[['suv', 'itemId']].drop_duplicates('suv').reset_index(drop=True)
tfidf_df['itemId'] = tfidf_df['itemId'].apply(lambda x:[str(i) for i in x])

tfidf_df['itemId_temp'] = tfidf_df['itemId'].apply(lambda x:' '.join(x))

print('=========Tfidf========')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.decomposition import TruncatedSVD

tfidf   = TfidfVectorizer(max_df=0.95, min_df=5, ngram_range=(1, 3))
res     = tfidf.fit_transform(tfidf_df['itemId_temp']) 

n_components = 32
print('svd start')
svd     = TruncatedSVD(n_components=n_components, n_iter=10, random_state=2022)
svd_res = svd.fit_transform(res)
print('svd finished')

tfidf_feats = []
for i in (range(n_components)):
    tfidf_df['groupby_suv_tfidf_svd_{}'.format(i)] = svd_res[:, i]
    tfidf_df['groupby_suv_tfidf_svd_{}'.format(i)] = tfidf_df['groupby_suv_tfidf_svd_{}'.format(i)].astype('float32')
    tfidf_feats.append('groupby_suv_tfidf_svd_{}'.format(i))


del tfidf_df['itemId_temp'],tfidf,res,svd,svd_res
gc.collect()

rec_itemId_tfidf = tfidf_df[['suv'] + tfidf_feats].sort_values('suv')[tfidf_feats].values
path = f'../feats/rec_seq_tfidf_size{n_components}_groupby_suv.npy'
np.save(path, rec_itemId_tfidf)



