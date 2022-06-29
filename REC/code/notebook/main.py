import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import random
import gc
import re
import time
import os

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

import pickle
def load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
seed_everything(2022)

version = 'B_提交'

X_test = load(f'../tmp/version{version}_X_test.pickle')
X_train = load(f'../tmp/version{version}_X_train.pickle')
y_train = load(f'../tmp/version{version}_y_train.pickle')

EMBEDDING = dict()

n_components = 32

hist_itemId_Seq_tfidf_path = f'../feats/hist_click_seq_tfidf_size{n_components}_groupby_suv.npy'
hist_itemId_Seq_tfidf = np.load(hist_itemId_Seq_tfidf_path)
EMBEDDING['hist_click_seq_tfidf_groupby_suv'] = hist_itemId_Seq_tfidf

rec_itemId_tfidf_groupby_pvId_path = f'../feats/rec_seq_tfidf_size{n_components}_groupby_pvId.npy'
rec_itemId_tfidf_groupby_pvId = np.load(rec_itemId_tfidf_groupby_pvId_path)
EMBEDDING['rec_seq_tfidf_groupby_pvId'] = rec_itemId_tfidf_groupby_pvId


rec_itemId_tfidf_groupby_suv_path = f'../feats/rec_seq_tfidf_size{n_components}_groupby_suv.npy'
rec_itemId_tfidf_groupby_suv = np.load(rec_itemId_tfidf_groupby_suv_path)
EMBEDDING['rec_seq_tfidf_groupby_suv'] = rec_itemId_tfidf_groupby_suv

####  itemId
W2V_SIZE = 64
itemId_w2v_path = f'../feats/itemId_embedding{W2V_SIZE}_window21_mincount5.npy'
itemId_w2v = np.load(itemId_w2v_path)
EMBEDDING['itemId_w2v'] = itemId_w2v

MAX_SEQUENCE_LENGTH = 21
SPARSE_nunique = load('../tmp/SPARSE_nunique.pickle')
SPARSE_FEATS = list(SPARSE_nunique.keys())

def attention_3d_block(inputs, seq_len=21):
    # input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(seq_len, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

def DIN(inputs):
    keys, query = inputs[:2]
    query = Lambda(lambda x: K.tile(x, [1, MAX_SEQUENCE_LENGTH]))(query)
    query = Lambda(lambda x: K.reshape(x, [-1, MAX_SEQUENCE_LENGTH, W2V_SIZE]))(query)
    subtracted = Subtract()([query, keys])
    multiplyed = Multiply()([query, keys])
    concat = Concatenate(axis=-1)([query, keys, subtracted, multiplyed])
    fc = Dense(80, activation='swish', use_bias=False)(concat)
    fc = Dense(40, activation='swish', use_bias=False)(fc)
    att_output = Dense(1, activation=None, use_bias=False)(fc)
    att_output = Lambda(lambda x: x / (W2V_SIZE ** 0.5))(att_output)
    att_output = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(att_output)
    att_output = Softmax()(att_output)
    output = Dot(axes=(2, 1))([att_output, keys])
    output = Reshape((W2V_SIZE,))(output)
    return output
    
def DeepFM_DIN(steps, seq_len=15, 
               att=True, lr=0.001, alpha=0.75
              ):
    
    
    ############################ 相关embedding ############################## 
    embedder_hist_suv = Embedding(
        EMBEDDING['hist_click_seq_tfidf_groupby_suv'].shape[0], 
        EMBEDDING['hist_click_seq_tfidf_groupby_suv'].shape[1],
        weights=[EMBEDDING['hist_click_seq_tfidf_groupby_suv']],
        trainable=False
    )
    
    embedder_rec_pvId = Embedding(
        EMBEDDING['rec_seq_tfidf_groupby_pvId'].shape[0], 
        EMBEDDING['rec_seq_tfidf_groupby_pvId'].shape[1],
        weights=[EMBEDDING['rec_seq_tfidf_groupby_pvId']],
        trainable=False
    )
    
    embedder_rec_suv = Embedding(
        EMBEDDING['rec_seq_tfidf_groupby_suv'].shape[0], 
        EMBEDDING['rec_seq_tfidf_groupby_suv'].shape[1],
        weights=[EMBEDDING['rec_seq_tfidf_groupby_suv']],
        trainable=False
    )
    

    ## embedder_itemId
    embedder_itemIds_w2v = Embedding(
        EMBEDDING['itemId_w2v'].shape[0],
        EMBEDDING['itemId_w2v'].shape[1],
        weights=[EMBEDDING['itemId_w2v']],
        trainable=False
    )

    ## pos embedding
    embedder_pos = Embedding(SPARSE_nunique['pos'], 16)
    
    ############################ suv pvId embedding ##############################
    suv_input = Input(shape=[1], dtype='int32')
    pvId_input = Input(shape=[1], dtype='int32')

    hist_suv_embed = embedder_hist_suv(suv_input)
    hist_suv_embed = Reshape((EMBEDDING['hist_click_seq_tfidf_groupby_suv'].shape[1],))(hist_suv_embed)
    
    rec_suv_embed = embedder_rec_suv(suv_input)
    rec_suv_embed = Reshape((EMBEDDING['rec_seq_tfidf_groupby_suv'].shape[1],))(rec_suv_embed)
        
    rec_pvId_embed = embedder_rec_pvId(pvId_input)
    rec_pvId_embed = Reshape((EMBEDDING['rec_seq_tfidf_groupby_pvId'].shape[1],))(rec_pvId_embed)
    
    ############################ DIN ############################## 
    ### itemIds_embed
    itemIds_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    itemIds_embed = embedder_itemIds_w2v(itemIds_input)
    
    ### itemId_embed w2v
    itemId_input = Input(shape=[1], dtype='int32')
    itemId_embed_ = embedder_itemIds_w2v(itemId_input)
    itemId_embed = Reshape((W2V_SIZE,))(itemId_embed_)
    
    
    dinLayer = DIN([itemIds_embed, itemId_embed])

    ############################ LSTM input  gap pos itemID ############################## 
    lstm_input = Input(shape=(seq_len, 3), name='lstm_input')
    
    lstm_input_gap = Lambda(lambda x: x[:, :, 0:1])(lstm_input)
#     lstm_input_logTs = Lambda(lambda x: x[:, :, 1:2])(lstm_input)
    
    lstm_input_pos = Lambda(lambda x: x[:, :, 1])(lstm_input)
    lstm_input_pos_emb = embedder_pos(lstm_input_pos)
    lstm_input_itemID = Lambda(lambda x: x[:, :, 2])(lstm_input)
    lstm_input_itemID_emb = embedder_itemIds_w2v(lstm_input_itemID)
    
    input_lstm_concat = Concatenate(axis=-1)([lstm_input_gap, 
#                                               lstm_input_logTs,
                                              lstm_input_pos_emb,
                                              lstm_input_itemID_emb])
    
    if att:
        lstm_out = Bidirectional(GRU(units=64, return_sequences=True))(input_lstm_concat)
        lstm_out = attention_3d_block(lstm_out, seq_len)
        lstm_out = Bidirectional(GRU(units=64, return_sequences=True))(lstm_out)
    else:
        lstm_out = Bidirectional(GRU(units=64, return_sequences=True))(input_lstm_concat)
        lstm_out = Bidirectional(GRU(units=64, return_sequences=True))(lstm_out)
        
    lstm_avg_pool = GlobalAveragePooling1D()(lstm_out)
    lstm_max_pool = GlobalMaxPooling1D()(lstm_out)
    lstm_conc = concatenate([lstm_avg_pool, lstm_max_pool])
    lstm_out = Dense(256)(lstm_conc)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Activation(activation="relu")(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)

    ############################ LSTM itemIds_embed ##############################
    ### LSTM
    concat_ = Concatenate(axis=1)([itemId_embed_, itemIds_embed])
    x = SpatialDropout1D(0.05)(concat_)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    x = Dense(256)(conc)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    itemIds_output = Dropout(0.2)(x)
    ### itemIds_avg_pool
    itemIds_avg_pool = GlobalAveragePooling1D()(itemIds_embed)
    
    ############################ DeepFM ##############################
    dense_input = Input((41,), name='dense_input', dtype='float32')
    # dense 特征的FM一阶部分
    fm_concat_dense_inputs = dense_input
    fst_order_dense_layer = Dense(1)(dense_input)
    
    # sparse 输入层
    sparse_inputs = []
    for f in SPARSE_FEATS:
        _input = Input([1], name=f)
        sparse_inputs.append(_input)
    
    # sparse FM一阶部分
    sparse_1d_embed = []
    for i, _input in enumerate(sparse_inputs):
        voc_size = SPARSE_nunique[SPARSE_FEATS[i]]
        _embed = Flatten()(Embedding(voc_size, 1)(_input))
        sparse_1d_embed.append(_embed)
    fst_order_sparse_layer = Add()(sparse_1d_embed)
    
    # 建立FM二阶部分
    k = 16
    ## sparse部分的二阶交叉
    sparse_kd_embed = []
    for i, _input in enumerate(sparse_inputs):
        voc_size = SPARSE_nunique[SPARSE_FEATS[i]]
        _embed = Embedding(voc_size, k)(_input)
        sparse_kd_embed.append(_embed)
        
    ## 1. 将所有的sparse的embedding拼接起来，得到(?, n, k)矩阵，n为特征数，k为embeddings_size
    concat_kd_embed = Concatenate(axis=1)(sparse_kd_embed)
    
    ## 2. axis=1列向求和再平方
    sum_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(concat_kd_embed)
    square_sum_kd_embed = Multiply()([sum_kd_embed, sum_kd_embed])  # ?, k
    
    ## 3. 先平方在求和
    square_kd_embed = Multiply()([concat_kd_embed, concat_kd_embed])
    sum_square_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(square_kd_embed) 
    
    ## 4. 相减除以2
    sub = Subtract()([square_sum_kd_embed, sum_square_kd_embed])
    sub = Lambda(lambda x: x*0.5)(sub)
    snd_order_layer = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(sub)
    
    # FM线性部分相加
    linear_part = Add()([fst_order_dense_layer, fst_order_sparse_layer])
    
    
    
    ############################ fc layer ############################
    flatten_embed = Flatten()(concat_kd_embed)  
    DNN_input = Concatenate(axis=1, name='dnn_input')([flatten_embed, 
                                                       
                                                       dense_input,
                                                       hist_suv_embed,
                                                       rec_pvId_embed,
                                                       rec_suv_embed,
                                                       
                                                       itemIds_output, 
                                                       dinLayer, 
                                                       lstm_out, 
                                                       itemIds_avg_pool,
                                                      ])
    x = BatchNormalization()(DNN_input)
    x = Dense(1024, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='swish')(x) 
    
    fc_output_layer = Dense(1)(x)
    output_layer = Add()([linear_part, snd_order_layer, fc_output_layer])
    output_layer = Activation('sigmoid')(output_layer)
    
    DeepFM = Model([dense_input]+sparse_inputs+[lstm_input, itemIds_input, 
                                                itemId_input, suv_input, pvId_input], output_layer)
     
    
    opt = tf.keras.optimizers.Adam(learning_rate = lr)
    DeepFM.compile(optimizer=opt,
                   loss='binary_crossentropy',
                   metrics=['binary_crossentropy', tf.keras.metrics.AUC(name='auc')]
                  )
    return DeepFM

def make_dataset(train, y, trn_idx, val_idx):
    X_trn, X_val = [], []
    for i in train:
        X_trn.append(i[trn_idx])
        X_val.append(i[val_idx])
    y_trn = y[trn_idx]
    y_val = y[val_idx]
    return X_trn, y_trn, X_val, y_val

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=2021)

oof = np.zeros([len(X_train[0]), 1])
predictions = np.zeros([len(X_test[0]), 1])
losses = []
aucs = []
bestEpochs = []
BATCH_SIZE = 1024
EPOCHS = 20

for fold_, (trn_idx, val_idx) in enumerate(folds.split(y_train, y_train)):
    
    print(f"fold n{fold_ + 1}")
    print("preparing dataset")
    X_trn, y_trn, X_val, y_val = make_dataset(X_train, y_train, trn_idx, val_idx)
    
    K.clear_session()
    n_training_rows = X_trn[0].shape[0]
    STEPS_PER_EPOCH = n_training_rows // BATCH_SIZE
    
    model = DeepFM_DIN(STEPS_PER_EPOCH*EPOCHS, 
                       seq_len=15, 
                       att=True, 
                       lr=0.001,
                       alpha = 0.75,
                      )
    if fold_ == 0:
        model.summary()

    if not os.path.exists(f"../model/version{version}"):
        os.makedirs(f"../model/version{version}") 
    if not os.path.exists(f"../result/version{version}"):
        os.makedirs(f"../result/version{version}")
        
    ckp_path = f"../model/version{version}/DeepFM_DIN_version{version}_fold{fold_}.h5"
    es = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_auc', min_delta = 1e-4, 
        patience = 5, mode = 'max')
    ckp = ModelCheckpoint(
        ckp_path, 
        monitor = 'val_auc', 
        verbose = 2,
        save_best_only = True,
        save_weights_only = True, 
        mode = 'max', 
        save_freq = 'epoch'
    )
       
    print('start training')
    history = model.fit(X_trn, y_trn,
              validation_data = (X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
              callbacks=[es, 
                         ckp, 
                        ]
             )
    
    hist = pd.DataFrame(history.history)
    
    loss = hist['val_binary_crossentropy'].min()
    auc = hist['val_auc'].max()
    best_epoch = hist[hist['val_auc'] == auc].index[0] + 1
    
    
    losses.append(loss)
    aucs.append(auc)
    bestEpochs.append(best_epoch)
    
    loss_path = f'../model/version{version}/DeepFM_DIN_version{version}_loss'
    bestEpoch_path = f'../model/version{version}/DeepFM_DIN_version{version}_best_epochs'
    auc_path = f'../model/version{version}/DeepFM_DIN_version{version}_auc'

    np.save(loss_path, losses)
    np.save(bestEpoch_path, bestEpochs)
    np.save(auc_path, aucs)

    model.load_weights(ckp_path)
    
    print('infering......')
    oof[val_idx] = model.predict(X_val)
    X_test_pred = model.predict(X_test)
    np.save(f'../result/version{version}/DeepFM_DIN_version{version}_fold{fold_}.npy', X_test_pred)
    del model, X_trn, y_trn, X_val, y_val;
    gc.collect()
    
np.save(f'../result/version{version}/DeepFM_DIN_version{version}_oof.npy', oof)    
predictions /= folds.n_splits
np.save(f'../result/version{version}/DeepFM_DIN_version{version}_predictions.npy', predictions) 

result = []
for fold_ in range(8):
    result.append(np.load(f"../result/version{version}/DeepFM_DIN_version{version}_fold{fold_}.npy"))
    
oof = np.load(f'../result/version{version}/DeepFM_DIN_version{version}_oof.npy')    
predictions = np.mean(result, axis=0)

test = pd.read_csv('../../data/Sohu2022_data/rec_data/test-dataset_B.csv')
test.rename(columns={'testSampleId':'sampleId'},inplace=True)
train['predict'] = oof.reshape(-1,)
train['label_nunique'] = train.groupby('pvId')['label'].transform('nunique')

gAUC = train[train['label_nunique'] > 1].groupby('pvId').parallel_apply(lambda x: roc_auc_score(x['label'], x['predict'])).mean()
submit = test[['sampleId']]
submit['result'] = predictions.reshape(-1,)
submit.rename(columns={'sampleId':'Id'}, inplace=True)
submit[['Id', 'result']].to_csv(f'../result/section2.txt', 
                                sep='\t', 
                                index=None)


