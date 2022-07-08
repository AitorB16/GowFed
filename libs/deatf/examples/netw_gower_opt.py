"""
This is the simplest use case of DEATF. 

In this instance, we require a simple DNN, a Multi Layer Perceptron (MLP). Only restrictions
for the evolution have to be established, like maximum number of layers or neurons in the MLP.
As is it the simple case, no evalution function has to be used, a predifined one is used (XEntropy).
Fashion mnist dataset is used, that is why 28x28 is the input size and 10 the output size.
"""
#import sys
#sys.path.append('..')
#sys.path.append("/home/tester/Desktop/TF/gower/gower")

import gower as gd
import pandas as pd


import numpy as np

from deatf.auxiliary_functions import load_fashion
from deatf.network import MLPDescriptor
from deatf.evolution import Evolving

from sklearn.preprocessing import OneHotEncoder

TRAIN_SIZE = 5000
TEST_SIZE = 500

if __name__ == "__main__":
  
    df = pd.read_csv("../datasets/TON_IoT-Datasets/Train_Test_datasets/Train_Test_Network_dataset/Train_Test_Network.csv")
    df.pop('type')
    df.pop('ts')
    df.head()

    cat_indexs = [0, 1, 2, 3, 4, 5, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 41]
    num_indexs = [6, 7, 8, 10, 11, 12, 13, 14, 33, 34]

    # Which cols are categorical
    cat_index_bool = [False] * 42
    for e in cat_indexs:
        cat_index_bool[e] = True

    data = df.sample(TRAIN_SIZE + TEST_SIZE, random_state=21)
    train_data = data.head(TRAIN_SIZE)
    train_labels = train_data.pop('label')
    test_data = data.tail(TEST_SIZE)
    test_labels = test_data.pop('label')
    labels = data.pop('label')

    gower_mat = gd.gower_matrix_limit_cols(data,TEST_SIZE,cat_features=cat_index_bool)

    train_gower_mat = gower_mat[:TRAIN_SIZE,:]

    test_gower_mat = gower_mat[TRAIN_SIZE:,:]


    #x_train, y_train, x_test, y_test, x_val, y_val = load_fashion()

    x_train = train_gower_mat
    y_train = train_labels

    x_test = test_gower_mat
    y_test = test_labels

    x_val = x_test
    y_val = y_test

    x_train = np.reshape(x_train, (-1, TEST_SIZE))
    x_test = np.reshape(x_test, (-1, TEST_SIZE))
    x_val = np.reshape(x_val, (-1, TEST_SIZE))

    OHEnc = OneHotEncoder()


    y_train = OHEnc.fit_transform(np.reshape(np.array(y_train), (-1, 1))).toarray()
    y_test = OHEnc.fit_transform(np.reshape(np.array(y_test), (-1, 1))).toarray()
    y_val = OHEnc.fit_transform(np.reshape(np.array(y_val), (-1, 1))).toarray()

   
    e = Evolving(evaluation="XEntropy", desc_list=[MLPDescriptor], compl=False,
                 x_trains=[x_train], y_trains=[y_train], x_tests=[x_val], y_tests=[y_val], 
                 n_inputs=[[TEST_SIZE]], n_outputs=[[2]],
                 population=10, generations=10, batch_size=200, iters=100, 
                 lrate=0.0001, cxp=0, mtp=1, seed=0,
                 max_num_layers=10, max_num_neurons=100, max_filter=4, max_stride=3,
                 evol_alg='mu_plus_lambda', sel='tournament', sel_kwargs={'tournsize':3}, 
                 evol_kwargs={}, batch_norm=False, dropout=False)
    
    a = e.evolve()
    print(a)
