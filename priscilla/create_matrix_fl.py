import numpy as np
import pandas as pd
import gower as gd
import configparser
import os

config_obj = configparser.ConfigParser()
config_obj.read('fl.ini')

init = config_obj['SETUP']

TRAIN_SIZE = int(init['train_size'])
TEST_SIZE = int(init['test_size'])
BALANCE_DATA = bool(int(init['balance_data']))
SEED = int(init['seed'])
NUM_CLIENTS = int(init['num_clients'])
RUN_NAME = init['run_name']

np.random.seed(SEED)


path = 'mats/fl/' + RUN_NAME + '/'
if not os.path.exists(path):
  os.mkdir(path)

#df = pd.read_csv("/home/abelenguer/scratch/projects/FL/TF/datasets/TON_IoT-Datasets/Train_Test_datasets/Train_Test_Network_dataset/Train_Test_Network.csv")
df = pd.read_csv('datasets/TON_IoT-Datasets/Train_Test_datasets/Train_Test_Network_dataset/Train_Test_Network.csv')
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

#Split data among clients
client_id_train = np.random.choice(range(NUM_CLIENTS), size=TRAIN_SIZE, replace=True)
client_id_test = np.random.choice(range(NUM_CLIENTS), size=TEST_SIZE, replace=True)

#Determine max_client_ds_size and min_client_ds_size
max_client_ds_size = -1
min_client_ds_size = len(train_data.loc[client_id_train == 0])
for id in range(0,NUM_CLIENTS):
    tmp_len = len(train_data.loc[client_id_train == id])
    if(tmp_len > max_client_ds_size):
        max_client_ds_size = tmp_len
    if(tmp_len < min_client_ds_size):
        min_client_ds_size = tmp_len



for id in range(0,NUM_CLIENTS):    
    tmp_features = train_data.loc[client_id_train == id]
    tmp_labels = train_labels.loc[client_id_train == id]
    tmp_gower_mat = np.matrix(gd.gower_matrix_limit_cols(tmp_features,min_client_ds_size,cat_features=cat_index_bool))
      
    with open(path + str(id) +'_train.csv','wb') as f:
        for line in tmp_gower_mat:
            np.savetxt(f, line, fmt='%.2f')
    
    with open(path + str(id) +'_train_labls.csv','wb') as f:
        np.savetxt(f, tmp_labels.values, fmt='%d')

    
for id in range(0,NUM_CLIENTS):    
    train_instances = train_data.loc[client_id_train == id]
    test_instances = test_data.loc[client_id_test == id]
    tmp_features = pd.concat([train_instances, test_instances])
    tmp_labels = test_labels.loc[client_id_test == id]
    tmp_gower_mat = np.matrix(gd.sliced_gower_matrix_limit_cols(tmp_features,min_client_ds_size,len(train_instances),cat_features=cat_index_bool))
    #tmp_gower_mat = np.matrix(tmp_gower_mat[len(train_instances):,:])

    with open(path + str(id) +'_test.csv','wb') as f:
        for line in tmp_gower_mat:
            np.savetxt(f, line, fmt='%.2f')

    with open(path + str(id) +'_test_labls.csv','wb') as f:
        np.savetxt(f, tmp_labels.values, fmt='%d')

with open(path + 'mat.ini', 'w') as f:
    out = '[MATX]\nseed = ' + str(SEED) + '\ntotal_train_size = ' + str(TRAIN_SIZE) + '\ntotal_test_size = ' + str(TEST_SIZE) + '\nmin_ds_client = ' + str(min_client_ds_size) + '\nmax_ds_client = ' + str(max_client_ds_size) + '\nnum_clients = ' + str(NUM_CLIENTS)
    f.write(out)