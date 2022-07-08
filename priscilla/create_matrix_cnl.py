import numpy as np
import pandas as pd
import gower as gd
import configparser
import os

config_obj = configparser.ConfigParser()
config_obj.read('cnl.ini')

init = config_obj['SETUP']

TRAIN_SIZE = int(init['train_size'])
TEST_SIZE = int(init['test_size'])
BALANCE_DATA = bool(int(init['balance_data']))
SEED = int(init['seed'])
RUN_NAME = init['run_name']

np.random.seed(SEED)

path = 'mats/cnl/' + RUN_NAME + '/'
if not os.path.exists(path):
  os.mkdir(path)

#df = pd.read_csv("/home/abelenguer/scratch/projects/FL/TF/datasets/TON_IoT-Datasets/Train_Test_datasets/Train_Test_Network_dataset/Train_Test_Network.csv")
df = pd.read_csv('../datasets/TON_IoT-Datasets/Train_Test_datasets/Train_Test_Network_dataset/Train_Test_Network.csv')
df.pop('type')
df.pop('ts')
#df.head()

# Percentage malware
perc = len(df.loc[df['label']==1])/len(df)
#print(perc)

# Balance dataset
if BALANCE_DATA:
    num_anom = len(df.loc[df['label']==1])
    df_anom = df.loc[df['label']==1]
    df_balance = df.loc[df['label']==0]
    df_balance = df_balance.sample(num_anom, replace=False, random_state = SEED)
    df_concated = pd.concat([df_balance, df_anom])
    balanced_data = df_concated
    df = balanced_data

data = df.sample(TRAIN_SIZE + TEST_SIZE, random_state = SEED)


cat_indexs = [0, 1, 2, 3, 4, 5, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 41]
num_indexs = [6, 7, 8, 10, 11, 12, 13, 14, 33, 34]
#bool_indexs = []

# Which cols are categorical
cat_index_bool = [False] * 42
for e in cat_indexs:
    cat_index_bool[e] = True


train_data = data.head(TRAIN_SIZE)
train_labels = train_data.pop('label')
test_data = data.tail(TEST_SIZE)
test_labels = test_data.pop('label')

#train_gower_mat = np.matrix(gd.gower_matrix_limit_cols(train_data,TRAIN_SIZE,cat_features=cat_index_bool))
train_gower_mat = np.matrix(gd.gower_matrix(train_data, cat_features=cat_index_bool))
test_gower_mat = np.matrix(gd.sliced_gower_matrix_limit_cols(pd.concat([train_data,test_data]),TRAIN_SIZE, TRAIN_SIZE,cat_features=cat_index_bool))
#test_gower_mat = np.matrix(test_gower_mat[TRAIN_SIZE:,:])


with open(path + 'train.csv','wb') as f:
    for line in train_gower_mat:
        np.savetxt(f, line, fmt='%.6f')

with open(path + 'train_labls.csv', 'wb') as f:
  np.savetxt(f, train_labels.values, fmt='%d')

with open(path + 'test.csv','wb') as f:
    for line in test_gower_mat:
        np.savetxt(f, line, fmt='%.6f')

with open(path + 'test_labls.csv', 'wb') as f:
  np.savetxt(f, test_labels.values, fmt='%d')

with open(path + 'mat.ini', 'w') as f:
    out = '[MATX]\nseed = ' + str(SEED) + '\ntrain_size = ' + str(TRAIN_SIZE) + '\ntest_size = ' + str(TEST_SIZE)
    f.write(out)