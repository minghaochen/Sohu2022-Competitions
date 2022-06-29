import pandas as pd
import numpy as np
import time
from pandarallel import pandarallel
from sklearn.preprocessing import StandardScaler, LabelEncoder
pandarallel.initialize()

## 1.1 - 1.3号
train = pd.read_csv('../../data/Sohu2022_data/rec_data/train-dataset_A.csv')
## 1.4
test_A = pd.read_csv('../../data/Sohu2022_data/rec_data/test-dataset_A.csv')
test_A.rename(columns={'testSampleId':'sampleId'},inplace=True)

## 1.5号
train_sup = pd.read_csv('../../data/Sohu2022_data/rec_data/train-dataset_sup.csv')

## 1.6
test_B = pd.read_csv('../../data/Sohu2022_data/rec_data/test-dataset_B.csv')
test_B.rename(columns={'testSampleId':'sampleId'},inplace=True)

train['month_day'] = (train['logTs']
                          .parallel_apply(lambda x:time.strftime("%m-%d", time.localtime(x/1000 + 28800))))
test_A['month_day'] = (test_A['logTs']
                          .parallel_apply(lambda x:time.strftime("%m-%d", time.localtime(x/1000 + 28800))))
train_sup['month_day'] = (train_sup['logTs']
                          .parallel_apply(lambda x:time.strftime("%m-%d", time.localtime(x/1000 + 28800))))
test_B['month_day'] = (test_B['logTs']
                          .parallel_apply(lambda x:time.strftime("%m-%d", time.localtime(x/1000 + 28800))))


userSeqB = test_B[['suv', 'userSeq']].drop_duplicates(['suv']).reset_index(drop=True)
userSeqB.fillna('-1', inplace=True)
userSeq_sup = train_sup[['suv', 'userSeq']].drop_duplicates(['suv']).reset_index(drop=True)
userSeq_sup.fillna('-1', inplace=True)
userSeq_test = pd.concat([userSeq_sup, userSeqB])
userSeq_test.drop_duplicates(['suv'], keep='last', inplace=True)
userSeq_dict = dict(zip(userSeq_test['suv'].values, userSeq_test['userSeq'].values))

def getLabel(row):
    suv = row.suv
    itemId = row.itemId
    label = np.nan
    if (suv in userSeq_dict):
        if str(itemId) in userSeq_dict[suv]:
            label = 1
        else:
            label = 0
    return label
test_A['label'] = test_A.parallel_apply(lambda x: getLabel(x), axis=1)

train['flag'] = 'trainA'
test_A['flag'] = 'testA'
train_sup['flag'] = 'train_sup'
test_B['flag'] = 'testB'

data = pd.concat([train, test_A, train_sup, test_B]).reset_index(drop=True)

cates = [
    'pvId', 'suv', 'operator', 'browserType', 
    'deviceType', 'osType', 'province', 'city'
]

for feat in cates:
    le = LabelEncoder()
    data[feat] = le.fit_transform(data[feat])
    
le = LabelEncoder()
data['userSeqFlag'] = le.fit_transform(data['userSeq'])

userSeq = (data[['suv', 'userSeq', 'userSeqFlag', 'logTs']]
           .drop_duplicates(['suv', 'userSeq'])
           .reset_index(drop=True))

userSeq = userSeq.sort_values('suv').reset_index(drop=True)

userSeq.to_csv('../../data/Sohu2022_data/rec_data/userSeqAll.csv', index=None)

del data['userSeq']

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

del data['sampleId']
data.reset_index(inplace=True)
data.rename(columns={'index': 'sampleId'}, inplace=True)


data = reduce_mem_usage(data)

data.to_feather('../../data/Sohu2022_data/rec_data/dataB.feather')