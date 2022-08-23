#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import gower as gd
import configparser
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys
#sys.path.append("/home/tester/Desktop/TF/federated/tensorflow_federated/examples/simple_fedavg")
#import simple_fedavg_tff

#config = tf.compat.v1.ConfigProto(gpu_options = 
#                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
## device_count = {'GPU': 1}
#)
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(session)

root = '/home/cali/Escritorio/FL-IDS/priscilla/'

config_obj = configparser.ConfigParser()
config_obj.read(root + 'cnl' + sys.argv[1] + '.ini')

init = config_obj['SETUP']
RUN_NAME = 'C' + sys.argv[1] #init['run_name']
PRINT_SCR = bool(int(init['print_scr']))
TRAIN_SIZE = int(init['train_size'])
TEST_SIZE = int(init['test_size'])
EPOCHS = int(init['epochs'])
BATCH_SIZE = int(init['batch_size'])
LEARNING_RATE = float(init['learning_rate'])
BALANCE_DATA = bool(int(init['balance_data']))
OUTLIERS = init['outliers']
SEED = int(init['seed'])

config_obj1 = configparser.ConfigParser()
config_obj1.read(root + 'mats/cnl/' + RUN_NAME + '/mat.ini')
init1 = config_obj1['MATX']
SEED = int(init1['seed'])
TRAIN_SIZE = int(init1['train_size'])
TEST_SIZE = int(init1['test_size'])

np.random.seed(SEED)
tf.random.set_seed(SEED)

#path = '/home/abelenguer/scratch/projects/FL/TF/centralized/experiments/' + RUN_NAME + '.txt'
result_path = root + 'results/cnl_AE/' + RUN_NAME + '/'
if not os.path.exists(result_path):
  os.mkdir(result_path)


#Save cofiguration
CONFIG_STR = '[SETUP]\n--AE version-- \nrun_name = ' + RUN_NAME + '\ntrain_size = ' + str(TRAIN_SIZE) + '\ntest_size = ' + str(TEST_SIZE) + '\nepochs = ' + str(EPOCHS) + '\nbatch_size = ' + str(BATCH_SIZE) + '\nlearning_rate = ' + str(LEARNING_RATE) + '\nbalance_data = ' + str(BALANCE_DATA) + '\noutliers = '+ OUTLIERS +'\nseed = ' + str(SEED) + '\n'
with open(result_path + 'conf.ini', 'w') as f: #Should be XML?
    f.write(CONFIG_STR)

train_data = np.array(pd.read_csv(root + 'mats/cnl/' + RUN_NAME + '/' + 'train.csv', sep='\s+', header=None))
train_labels = pd.read_csv(root + 'mats/cnl/' + RUN_NAME + '/train_labls.csv', header=None)
test_data = np.array(pd.read_csv(root + 'mats/cnl/' + RUN_NAME + '/test.csv', sep='\s+', header=None))
test_labels = pd.read_csv(root + 'mats/cnl/' + RUN_NAME + '/test_labls.csv', header=None)

normal_train_data = train_data[~train_labels]
normal_test_data = test_data[~test_labels]

anomalous_train_data = train_data[train_labels]
#anomalous_test_data = test_data[test_labels]

def create_keras_model():
  initializer=tf.keras.initializers.GlorotUniform(seed= SEED)
  return tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(TRAIN_SIZE,)),
    tf.keras.layers.Dense(128, activation="relu", kernel_initializer=initializer),
    tf.keras.layers.Dense(64, activation="relu", kernel_initializer=initializer),
    tf.keras.layers.Dense(32, activation="relu", kernel_initializer=initializer),
    tf.keras.layers.Dense(16, activation="relu", kernel_initializer=initializer),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(8, activation="relu", kernel_initializer=initializer),
    tf.keras.layers.Dense(16, activation="relu", kernel_initializer=initializer),
    tf.keras.layers.Dense(32, activation="relu", kernel_initializer=initializer),
    tf.keras.layers.Dense(64, activation="relu", kernel_initializer=initializer),
    tf.keras.layers.Dense(128, activation="relu", kernel_initializer=initializer),
    tf.keras.layers.Dense(TRAIN_SIZE, activation="sigmoid", kernel_initializer=initializer)])


model = create_keras_model()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
#metrics = [tf.keras.metrics.BinaryAccuracy()]
loss = [tf.keras.losses.mae]
model.compile(optimizer=opt, loss=loss)#, metrics = metrics)

history = model.fit(train_data, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(normal_test_data, normal_test_data),
                    shuffle=True,
                    callbacks=[early_stopping], verbose=PRINT_SCR
                    )



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


reconstruction = model.predict(normal_train_data)
reconstruction_error = tf.keras.losses.mae(reconstruction, normal_train_data)
threshold = np.mean(reconstruction_error) + np.std(reconstruction_error)

reconstruction_test = model.predict(test_data)
reconstruction_error_test = tf.keras.losses.mae(reconstruction_test, test_data)
preds = tf.math.greater(reconstruction_error_test, threshold)


out = save_stats(np.round(preds, decimals=0), test_labels.astype(int), PRINT_SCR)

# save model
#model.save('/home/abelenguer/scratch/projects/FL/TF/centralized/experiments/' + RUN_NAME + '.h5')
model.save(result_path + 'model.h5')


# Save result
tr_loss = ''
val_loss = ''

for e in history.history["loss"]:
  tr_loss = tr_loss + str(e) + ' '

for e in history.history["val_loss"]:
  val_loss = val_loss + str(e) + ' '

with open(result_path + 'stats.txt', 'w') as f:
  f.write(out + " \n")

with open(result_path + 'tr_loss.txt', 'w') as f:
  f.write(tr_loss + "\n")

with open(result_path + 'val_loss.txt', 'w') as f:
  f.write(val_loss + "\n")

#first_layer_weights = model.layers[0].get_weights()[0]
#print(first_layer_weights)


