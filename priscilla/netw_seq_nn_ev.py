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

from ctypes.wintypes import INT
import gower as gd
import pandas as pd

import configparser

import numpy as np

from deatf.network import MLPDescriptor
from deatf.evolution import Evolving

from sklearn.preprocessing import OneHotEncoder

MODE = 'cnl' #fl

config_obj = configparser.ConfigParser()
config_obj.read(MODE + '.ini')

init = config_obj['SETUP']

RUN_NAME = init['run_name']
TRAIN_SIZE = int(init['train_size'])
TEST_SIZE = int(init['test_size'])
SEED = int(init['seed'])

if MODE == 'cnl':
    train_data = np.array(pd.read_csv('mats/cnl/' + RUN_NAME + '/train.csv', sep='\s+', header=None))
    train_labels = pd.read_csv('mats/cnl/' + RUN_NAME + '/train_labls.csv', header=None)
    test_data = np.array(pd.read_csv('mats/cnl/' + RUN_NAME + '/test.csv', sep='\s+', header=None))
    test_labels = pd.read_csv('mats/cnl/' + RUN_NAME + '/test_labls.csv', header=None)

x_train = train_data
y_train = train_labels

x_test = test_data
y_test = test_labels

x_val = x_test
y_val = y_test

#x_train = np.reshape(x_train, (-1, TRAIN_SIZE))
#x_test = np.reshape(x_test, (-1, TRAIN_SIZE))
#x_val = np.reshape(x_val, (-1, TRAIN_SIZE))

OHEnc = OneHotEncoder()
y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()
y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
y_val = OHEnc.fit_transform(np.reshape(y_val, (-1, 1))).toarray()

#e = Evolving(evaluation="XEntropy", desc_list=[MLPDescriptor], compl=False,
#             x_trains=[x_train], y_trains=[y_train], x_tests=[x_val], y_tests=[y_val], 
#             n_inputs=[[TRAIN_SIZE]], n_outputs=[[2]],
#             population=10, generations=10, batch_size=200, iters=150, 
#             lrate=0.0001, cxp=0, mtp=1, seed=SEED,
#             max_num_layers=10, max_num_neurons=100, max_filter=4, max_stride=3,
#             evol_alg='mu_plus_lambda', sel='tournament', sel_kwargs={'tournsize':3}, 
#             evol_kwargs={}, batch_norm=False, dropout=False, run_name=str(RUN_NAME))

e = Evolving(evaluation="XEntropy", desc_list=[MLPDescriptor], compl=False,
             x_trains=[x_train], y_trains=[y_train], x_tests=[x_val], y_tests=[y_val], 
             n_inputs=[[TRAIN_SIZE]], n_outputs=[[2]],
             population=10, generations=10, batch_size=200, iters=150, 
             lrate=0.0001, cxp=0, mtp=1, seed=SEED,
             max_num_layers=100, max_num_neurons=1000, max_filter=4, max_stride=3,
             evol_alg='mu_plus_lambda', sel='tournament', sel_kwargs={'tournsize':3}, 
             evol_kwargs={}, batch_norm=False, dropout=False, run_name=str(RUN_NAME))

a = e.evolve()
#print(a)
