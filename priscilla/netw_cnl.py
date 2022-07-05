#!/usr/bin/env python
# coding: utf-8

from random import seed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import gower as gd
import configparser
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys
sys.path.append("/home/tester/Desktop/TF/federated/tensorflow_federated/examples/simple_fedavg")
import simple_fedavg_tff

config_obj = configparser.ConfigParser()
config_obj.read('cnl.ini')

init = config_obj['SETUP']
RUN_NAME = init['run_name']
PRINT_SCR = bool(int(init['print_scr']))
TRAIN_SIZE = int(init['train_size'])
TEST_SIZE = int(init['test_size'])
EPOCHS = int(init['epochs'])
BATCH_SIZE = int(init['batch_size'])
LEARNING_RATE = float(init['learning_rate'])
BALANCE_DATA = bool(int(init['balance_data']))
SEED = int(init['seed'])

np.random.seed(SEED)
tf.random.set_seed(SEED)

#path = '/home/abelenguer/scratch/projects/FL/TF/centralized/experiments/' + RUN_NAME + '.txt'
path = 'results/cnl/' + RUN_NAME + '/'
if not os.path.exists(path):
  os.mkdir(path)


#Save cofiguration
CONFIG_STR = '[SETUP]\nrun_name = ' + RUN_NAME + '\ntrain_size = ' + str(TRAIN_SIZE) + '\ntest_size = ' + str(TEST_SIZE) + '\nepochs = ' + str(EPOCHS) + '\nbatch_size = ' + str(BATCH_SIZE) + '\nlearning_rate = ' + str(LEARNING_RATE) + '\nbalance_data = ' + str(BALANCE_DATA) + '\nseed = ' + str(SEED) + '\n'
with open(path + 'conf.ini', 'w') as f: #Should be XML?
    f.write(CONFIG_STR)


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
  num_anom = len(df.loc[df['label']==1.])
  df_anom = df.loc[df['label']==1.]
  df_normal = df.loc[df['label']==0.]
  df_normal = df_normal.sample(num_anom, replace=False, random_state = SEED)
  df_concated = pd.concat([df_normal, df_anom])
  balanced_data = df_concated
  df = balanced_data

data = df.sample(TRAIN_SIZE+TEST_SIZE, random_state= SEED)

cat_indexs = [0, 1, 2, 3, 4, 5, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 41]
num_indexs = [6, 7, 8, 10, 11, 12, 13, 14, 33, 34]
#bool_indexs = []

# Which cols are categorical
cat_index_bool = [False] * 42
for e in cat_indexs:
    cat_index_bool[e] = True

train_data = data.head(TRAIN_SIZE).copy()
train_labels = train_data.pop('label')
test_data = data.copy()
test_labels = test_data.pop('label')
test_labels = test_labels.tail(TEST_SIZE)


train_gower_mat = gd.gower_matrix_limit_cols(train_data,TEST_SIZE,cat_features=cat_index_bool)
test_gower_mat = gd.gower_matrix_limit_cols(test_data,TEST_SIZE,cat_features=cat_index_bool)
test_gower_mat = test_gower_mat[TRAIN_SIZE:,:]


def create_keras_model():
  initializer=tf.keras.initializers.GlorotUniform(seed= SEED)
  return tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(TEST_SIZE,)),
      tf.keras.layers.Dense(1024, activation="relu", kernel_initializer=initializer),
      tf.keras.layers.Dense(512, activation="relu", kernel_initializer=initializer),
      tf.keras.layers.Dense(256, activation="relu", kernel_initializer=initializer),
      tf.keras.layers.Dense(256, activation="relu", kernel_initializer=initializer),
      #tf.keras.layers.Dense(256, activation="relu"),
      #tf.keras.layers.Dense(256, activation="relu"),
      #tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dropout(0.2),
      #tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(128, activation="relu", kernel_initializer=initializer),
      tf.keras.layers.Dense(2, activation="relu", kernel_initializer=initializer),
      #tf.keras.layers.Dense(2, activation="relu"),
      #tf.keras.layers.Dense(32, activation="relu"),
      #tf.keras.layers.Dense(4, activation="relu"),
      tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer=initializer)
      #tf.keras.layers.Softmax()
  ])


train_data = train_gower_mat

model = create_keras_model()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
metrics = [tf.keras.metrics.BinaryAccuracy()]
loss = [tf.keras.losses.BinaryCrossentropy()]
model.compile(optimizer=opt, loss=loss, metrics = metrics)

history = model.fit(train_data, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(test_gower_mat, test_labels),
                    shuffle=True,
                    callbacks=[early_stopping], verbose=PRINT_SCR
                    )


#if PRINT_SCR:
#  plt.plot(history.history["loss"], label="Training Loss")
#  plt.plot(history.history["val_loss"], label="Validation Loss")
#  plt.yscale('log')
#  plt.legend()


#Results = model.evaluate(test_gower_mat, test_labels)
#print("test loss, test acc:", results)


def save_stats(predictions, labels, print_sc=False):

  acc = "{:.4f}".format(accuracy_score(labels, predictions))
  prec = "{:.4f}".format(precision_score(labels, predictions, zero_division=0))
  rcl = "{:.4f}".format(recall_score(labels, predictions, zero_division=0))
  f1 = "{:.4f}".format(f1_score(labels, predictions, zero_division=0))
  roc = "{:.4f}".format(roc_auc_score(labels, predictions))

  if print_sc:
    print("Accuracy = " + acc)
    print("Precision = " + prec)
    print("Recall = " + rcl) #sensitivity
    print("F1 = " + f1)
    print("ROC_AUC = " + roc)

  return (acc + " " + prec + " " + rcl + " " + f1 + " " + roc)


preds = model.predict(test_gower_mat)
out = save_stats(np.round(preds, decimals=0), test_labels.astype(int), PRINT_SCR)


# save model
#model.save('/home/abelenguer/scratch/projects/FL/TF/centralized/experiments/' + RUN_NAME + '.h5')
model.save(path + 'model.h5')


# Save result
tr_loss = ''
val_loss = ''

for e in history.history["loss"]:
  tr_loss = tr_loss + str(e) + ' '

for e in history.history["val_loss"]:
  val_loss = val_loss + str(e) + ' '

with open(path + 'stats.txt', 'w') as f:
  f.write(out + " \n")

with open(path + 'tr_loss.txt', 'w') as f:
  f.write(tr_loss + "\n")

with open(path + 'val_loss.txt', 'w') as f:
  f.write(val_loss + "\n")

#first_layer_weights = model.layers[0].get_weights()[0]
#print(first_layer_weights)



