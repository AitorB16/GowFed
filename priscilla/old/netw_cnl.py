#!/usr/bin/env python
# coding: utf-8



from tabnanny import verbose
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
#from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

#import sys
#sys.path.append("../gower/gower")
#import gower_dist as gd

import gower as gd


RUN_NAME = 'C2'
PRINT_SCR = True

TRAIN_SIZE = 1000
TEST_SIZE = 50

EPOCHS = 500
BATCH_SIZE = 64

df = pd.read_csv("../datasets/TON_IoT-Datasets/Train_Test_datasets/Train_Test_Network_dataset/Train_Test_Network.csv")
df.pop('type')
df.pop('ts')
#df.head()


# Percentage malware
perc = len(df.loc[df['label']==1])/len(df)
#print(perc)

# Balance dataset
num_anom = len(df.loc[df['label']==1.])
df_anom = df.loc[df['label']==1.]
df_normal = df.loc[df['label']==0.]
df_normal = df_normal.sample(num_anom, replace=False)
df_concated = pd.concat([df_normal, df_anom])
balanced_data = df_concated


data = balanced_data.sample(TRAIN_SIZE+TEST_SIZE, random_state=21)


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
  return tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(TEST_SIZE,)),
      tf.keras.layers.Dense(1024, activation="relu"),
      tf.keras.layers.Dense(512, activation="relu"),
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(256, activation="relu"),
      #tf.keras.layers.Dense(256, activation="relu"),
      #tf.keras.layers.Dense(256, activation="relu"),
      #tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dropout(0.2),
      #tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(128, activation="relu"),
      tf.keras.layers.Dense(2),
      #tf.keras.layers.Dense(2, activation="relu"),
      #tf.keras.layers.Dense(32, activation="relu"),
      #tf.keras.layers.Dense(4, activation="relu"),
      #tf.keras.layers.Dense(1, activation="sigmoid"),
      tf.keras.layers.Softmax()
  ])


train_data = train_gower_mat


model = create_keras_model()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
loss = [tf.keras.losses.SparseCategoricalCrossentropy()]
model.compile(optimizer=opt, loss=loss, metrics = metrics)


history = model.fit(train_data, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(train_data, train_labels),
                    shuffle=True,
                    callbacks=[early_stopping],
                    verbose=0
                    )


if(PRINT_SCR):
  plt.plot(history.history["loss"], label="Training Loss")
  plt.plot(history.history["val_loss"], label="Validation Loss")
  plt.yscale('log')
  plt.legend()


results = model.evaluate(test_gower_mat, test_labels)
#print("test loss, test acc:", results)


def save_stats(predictions, labels, print_scr=False):

  acc = "{:.4f}".format(accuracy_score(labels, predictions))
  prec = "{:.4f}".format(precision_score(labels, predictions, zero_division=0))
  rcl = "{:.4f}".format(recall_score(labels, predictions, zero_division=0))
  f1 = "{:.4f}".format(f1_score(labels, predictions, zero_division=0))
  roc = "{:.4f}".format(roc_auc_score(labels, predictions))

  if print_scr:
    print("Accuracy = " + acc)
    print("Precision = " + prec)
    print("Recall = " + rcl) #sensitivity
    print("F1 = " + f1)
    print("ROC_AUC = " + roc)

  return (acc + " " + prec + " " + rcl + " " + f1 + " " + roc)



preds = model.predict(test_gower_mat)
out = save_stats(np.round(preds[:,1], decimals=0), test_labels.astype(int), PRINT_SCR)

# save model
model.save('./experiments/' + RUN_NAME + '.h5')

# Save result in txt
path = './experiments/' + RUN_NAME + '.txt'
tr_loss = ''
val_loss = ''

for e in history.history["loss"]:
  tr_loss = tr_loss + str(e) + ' '

for e in history.history["val_loss"]:
  val_loss = val_loss + str(e) + ' '

with open(path, 'w') as f:
  f.write('ACC PREC RCL F1 ROC \n')
  f.write(out + " \n\n" + tr_loss + "\n\n" + val_loss + "\n")


  






