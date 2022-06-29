import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import random
import gc
import re
import time
import os
import pickle

from pandarallel import pandarallel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

from gensim.models.word2vec import Word2Vec

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import keras

pd.set_option('display.max_columns', None)

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

pandarallel.initialize()


def save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
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

# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
seed_everything(2022)

version = 'B_提交'

data = pd.read_feather('../../data/Sohu2022_data/rec_data/dataB.feather')
train = data[data.label.notnull()]
test = data[data.flag == 'testB']
data = pd.concat([train, test]).reset_index(drop=True)
data = data.reset_index(drop=True)
gc.collect()

###### base 统计特征 deepFM
base_static_path = f'../feats/static_feats.feather'
base_feats_df = pd.read_feather(base_static_path)
data = data.merge(base_feats_df, on='sampleId', how='left')
del base_feats_df; gc.collect()

###### time 统计特征 deepFM
time_path = f'../feats/time_feats.feather'
time_feats_df = pd.read_feather(time_path)
data = data.merge(time_feats_df, on='sampleId', how='left')
del time_feats_df; gc.collect()

ItemId_sorted_path = f'../feats/user_itemIdSeq_sorted.csv'
ItemId_sorted_df = pd.read_csv(ItemId_sorted_path)
ItemId_sorted_df['itemIdSeq_sorted'] = ItemId_sorted_df['itemIdSeq_sorted'].parallel_apply(eval)
data = data.merge(ItemId_sorted_df, on=['suv', 'userSeqFlag'], how='left')
del ItemId_sorted_df; gc.collect()

itmeId_click_path = f'../feats/itemId_hist_feats.feather'
itmeId_click = pd.read_feather(itmeId_click_path)
data = data.merge(itmeId_click, on=['suv', 'itemId', 'logTs'], how='left')
del itmeId_click; gc.collect()

###  用户历史点击序列 的统计特征 
# 各文章历史点击序列的 均值方差最大值最小值  推荐文章历史点击次数
seq_click_path = f'../feats/hist_seq_feats.feather'
seq_click = pd.read_feather(seq_click_path)

tmp_feats = []
for col in ['count', 'click']:
    for agg in [
        'std', 'mean', 
    ]:
        feat = f'hist_seq_{col}_{agg}' 
        tmp_feats.append(feat)
data = data.merge(seq_click[['suv'] + tmp_feats], on='suv', how='left')


for col in ['count', 'click']:
    for agg in [
        'std', 'mean', 
    ]:
        feat = f'hist_seq_{col}_{agg}'
        data[feat].fillna(seq_click[feat].astype(np.float64).mean(), inplace=True)
del seq_click; gc.collect() 

###  用户历史点击序列 的穿越统计特征 
itmeId_overall_click_path = f'../feats/itemId_overall_feats.feather'
itmeId_overall_click = pd.read_feather(itmeId_overall_click_path)
data = data.merge(itmeId_overall_click, on='itemId', how='left')
data['overall_itemId_click'].fillna(itmeId_overall_click.overall_itemId_click.mean(), inplace=True)

tmp_feats = []
for col in ['click']:
    for agg in [
        'std', 'mean', 
    ]:
        feat = f'overall_{col}_{agg}'
        tmp_feats.append(feat)
        
seq_overall_click_path = f'../feats/hist_seq_overall_feats.feather'
seq_overall_click = pd.read_feather(seq_overall_click_path)
data = data.merge(seq_overall_click[['suv'] + tmp_feats], on='suv', how='left')
data[tmp_feats].fillna(seq_overall_click[tmp_feats].mean(), inplace=True)
del itmeId_overall_click, seq_overall_click; gc.collect()

#######  pos(cate) + gap(value) + itmeId(cate)   LSTM input
pos_gap_path = f'../feats/pos_gap_feats_all_gapmixmin.feather'
pos_gap_feats_df = pd.read_feather(pos_gap_path)
data = data.merge(pos_gap_feats_df, on='sampleId', how='left')

### 情感特征
emotion_path = f'../feats/emotions_feats.feather'
emotion_feats = pd.read_feather(emotion_path)
data = data.merge(emotion_feats, on='sampleId', how='left')
del emotion_feats;gc.collect()

SPARSE_FEATS = [
    'new_itemId',
    'browserType', 
    'osType', 
    'city', 
    'operator', 
    'itemId_class',
    'pos',
]

DENSE_FEATS = [
    ####### base static feats
    'operator_count', 'browserType_count', 'deviceType_count',
    'osType_count', 'province_count', 'city_count',
    
    'itemId_count','suv_count', 'pvId_count', 
    'itemId_suv_nunique', 'suv_itemId_nunique','itemId_pvId_nunique', 
    'pvId_itemId_nunique', 'pvId_suv_nunique','suv_pvId_nunique',
    
    #####  time feats
    'time_last', 'time_first',
    'time_long', 'time_gap_last', 'time_gap_first', 
    'month_day_nunique', 
    'hour_nunique', 
    'time_diff_mean', 'time_diff_max','time_diff_min', 'time_diff_var', 
    
    #### click feats
    'hist_seq_count_std', 
    'hist_seq_count_mean',
    'hist_seq_click_std',
    'hist_seq_click_mean',    
    'hist_count', 'hist_click',
    
    ### 全局
    'overall_itemId_click',
    
    'overall_click_mean',
    'overall_click_std',
    
    'pos_rato',
    'pos_click',
    
    'pos_logTs_class_rato',
    'pos_logTs_class_click',
    
    ### gap
    'gap',
    'logTs',
    
    'hist_emotion_sim', 
    'recent3_emotion_sim',
]

l = 7
POS_GAP_FEATS = []
for i in range(l * 2 + 1):
    POS_GAP_FEATS += ['gap_%s' % (i - l),'pos_%s' % (i - l), 'itemId_%s' % (i - l)]


### sparse label unique
SPARSE_nunique = dict()
for feat in SPARSE_FEATS:
    if feat == 'pos': continue
    le = LabelEncoder()
    data[feat] = le.fit_transform(data[feat])
    SPARSE_nunique[feat] = data[feat].nunique() + 1

## pos
le = LabelEncoder()
le = le.fit(data['pos'] + data['pos_-7'])
SPARSE_nunique['pos'] = data['pos'].nunique() + 1
for feat in ['pos'] + [i for i in POS_GAP_FEATS if 'pos' in i]:
    data[feat] = le.fit_transform(data[feat])
save('../tmp/SPARSE_nunique.pickle', SPARSE_nunique)

maxmin = [
    'logTs',    
    'hist_emotion_sim', 
    'recent3_emotion_sim',
    ]

for feat in maxmin: 
    data[feat] = data.groupby('pvId')[feat].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    data[feat].fillna(0, inplace=True)
    
# 标准化
from sklearn import preprocessing
zscore = preprocessing.StandardScaler()
# 标准化处理
zscore_featuers = DENSE_FEATS.copy()

for f in zscore_featuers:
    if f in maxmin + ['gap', 'logTs']:
        zscore_featuers.remove(f)
        
data[zscore_featuers] = zscore.fit_transform(data[zscore_featuers])
data[zscore_featuers] = data[zscore_featuers].astype(np.float32)

from tensorflow.keras.preprocessing import text, sequence

haveConvert = False
MAX_SEQUENCE_LENGTH = 21

itemIdSeq_w2v = pd.read_csv('../tmp/w2v_itemIdSeq_sorted_train.csv')
itemIdSeq_w2v['itemIdSeq_sorted'] = itemIdSeq_w2v['itemIdSeq_sorted'].parallel_apply(lambda x: eval(x))
print(itemIdSeq_w2v['itemIdSeq_sorted'][1])
tokenizer = text.Tokenizer(num_words=None)
tokenizer.fit_on_texts(itemIdSeq_w2v['itemIdSeq_sorted'].values.tolist())
word_index = tokenizer.word_index

if not haveConvert:
    itemIdSeq_train = data[~data['label'].isna()]['itemIdSeq_sorted'].values.tolist()
    itemIdSeq_test = data[data['label'].isna()]['itemIdSeq_sorted'].values.tolist()
    print(itemIdSeq_train[1])
    itemIdSeq_train = tokenizer.texts_to_sequences(itemIdSeq_train)
    itemIdSeq_train = sequence.pad_sequences(itemIdSeq_train, maxlen=MAX_SEQUENCE_LENGTH)
    np.save(f'../feats/itemIdSeq_train_ItemId_sorted_convert_length{MAX_SEQUENCE_LENGTH}.npy', itemIdSeq_train)
    
    itemIdSeq_test = tokenizer.texts_to_sequences(itemIdSeq_test)
    itemIdSeq_test = sequence.pad_sequences(itemIdSeq_test, maxlen=MAX_SEQUENCE_LENGTH)
    np.save(f'../feats/itemIdSeq_test_ItemId_sorted_convert_length{MAX_SEQUENCE_LENGTH}.npy', itemIdSeq_test)
    
for f in ['itemId'] + [i for i in POS_GAP_FEATS if 'itemId' in i]:
    data[f] = data[f].astype(str)
    itemId = tokenizer.texts_to_sequences(data[f])
    itemId = sequence.pad_sequences(itemId, maxlen=1)
    itemId = np.reshape(itemId, (-1))
    data[f] = itemId
    del itemId; gc.collect()
    
W2V_SIZE = 64
window_size = 21
min_count = 5
w2v_model = Word2Vec.load(f'../model/hist_click_seq_w2v_size{W2V_SIZE}_window{window_size}_mincount{min_count}.model')

NUM_WORDS = len(word_index) + 1
print('Total %s word vectors.' % NUM_WORDS)

EMBEDDING_MATRIX = np.zeros((NUM_WORDS, W2V_SIZE))

for word, i in word_index.items():
    if (word in w2v_model.wv.key_to_index) :
        embedding_vector = w2v_model.wv[word]
    else:
        embedding_vector = np.random.random(W2V_SIZE) * 0.5
        embedding_vector = embedding_vector - embedding_vector.mean()
        
    EMBEDDING_MATRIX[i] = embedding_vector
    
np.save(f'../feats/itemId_embedding{W2V_SIZE}_window{window_size}_mincount{min_count}.npy', EMBEDDING_MATRIX)
del EMBEDDING_MATRIX; gc.collect()

df_train = data[['sampleId', 'label', 'itemId', 'suv', 'pvId'] + SPARSE_FEATS + DENSE_FEATS + POS_GAP_FEATS]
feats = [f for f in df_train.columns if f not in maxmin]
df_train[feats] = reduce_mem_usage(df_train[feats])

MAX_SEQUENCE_LENGTH = 21

itemIdSeq_train = np.load(f'../feats/itemIdSeq_train_ItemId_sorted_convert_length{MAX_SEQUENCE_LENGTH}.npy')
itemIdSeq_test = np.load(f'../feats/itemIdSeq_test_ItemId_sorted_convert_length{MAX_SEQUENCE_LENGTH}.npy')

train = df_train[~df_train['label'].isna()].reset_index(drop=True)
test = df_train[df_train['label'].isna()].reset_index(drop=True)

def make_dataset(data, itemIdSeq, mode='train'):
    dense_x = data[DENSE_FEATS].values
    sparse_x = [data[f].values for f in SPARSE_FEATS]
    lstm_x = data[POS_GAP_FEATS].values.reshape(len(data), int(len(POS_GAP_FEATS)/3), 3)
    itemIds_x = itemIdSeq
    itemId_x = data['itemId'].values
    suv_x = data['suv'].values
    pvId_x = data['pvId'].values
    y_categorical = None
    if mode == 'train':
        y_categorical = data['label'].values
    return [dense_x] + sparse_x + [lstm_x, itemIds_x, itemId_x, suv_x, pvId_x], y_categorical

X_test, _ = make_dataset(test, itemIdSeq_test, mode = 'test')
X_train, y_train = make_dataset(train, itemIdSeq_train, mode='train')

save(f'../tmp/version{version}_X_test.pickle', X_test)
save(f'../tmp/version{version}_X_train.pickle', X_train)
save(f'../tmp/version{version}_y_train.pickle', y_train)