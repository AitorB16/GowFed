import sys
sys.path.append('../..')

import tensorflow as tf
import numpy as np
import time

from deatf.auxiliary_functions import accuracy_error
from deatf.network import RNNDescriptor
from deatf.evolution import Evolving

from aux_functions_testing import load_dataset, select_evaluation

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as opt

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]

def test_RNN_all_datasets(eval_func=None, batch_size=150, population=5, 
                      is_time_series=True,series_input_width=30, series_label_width=1, 
                      generations=10, iters=100, max_num_layers=10, max_num_neurons=20,
                      evol_alg='mu_plus_lambda', sel='best', lrate=0.01, cxp=0, mtp=1,
                      seed=None, sel_kwargs={}, max_filter=4, max_stride=3, hyperparameters={}):
    """
    Tests the RNN network with all the possible datasets and with the specified parameter selection.

    :param dataset_name: Name of the dataset that will be used in the genetic algorithm.
    :param eval_func: Evaluation function for evaluating each network.
    :param batch_size: Batch size of the data during the training of the networks.
    :param population: Number of individuals in the populations in the genetic algorithm.
    :param generations: Number of generations that will be done in the genetic algorithm.
    :param iters: Number of iterations that each network will be trained.
    :param max_num_layers: Maximum number of layers allowed in the networks.
    :param max_num_neurons: Maximum number of neurons allowed in the networks.
    :param max_filter: Maximum size of the filter allowed in the networks.
    :param max_stride: Maximum size of the stride allowed in the networks.
    :param evol_alg: Evolving algorithm that will be used during the genetic algorithm.
    :param sel: Selection method that will be used during the genetic algorithm.
    :param sel_kwargs: Arguments for selection method.
    :param lrate: Learning rate that will be used during training.
    :param cxp: Crossover probability that will be used during the genetic algorithm.
    :param mtp: Mutation probability that will be used during the genetic algorithm.
    :param seed: Seed that will be used in every random method.
    :param hyperparameters: Hyperparameters that will be evolved during the genetic algorithm.
    :param is_time_series: Boolean that indicates if the data is a time series.
    :param series_input_width: Width of the input series data. 
    :param series_label_width: Width of the labels series data. 
    """  
    
    dataset_collection = ['mnist', 'kmnist', 'air_quality', 'estambul_values']
   
    for dataset in dataset_collection:
        
        print('\nEvaluating the {} dataset with the following configuration:'.format(dataset),
              '\nBatch size:  {}\nPopulation of networks:  {}\nGenerations:  {}'.format(batch_size, population, generations),
              '\nIterations in each network:  {}\nMaximum number of layers:  {}'.format(iters, max_num_layers),
              '\nMaximum number of neurons in each layer: {}'.format(max_num_neurons))

        init_time = time.time()
        
        try:
            x = test_RNN(dataset, is_time_series=is_time_series, 
                 series_input_width=series_input_width, series_label_width=series_label_width,
                 eval_func=eval_func, batch_size=batch_size, 
                 population=population, generations=generations, iters=iters, 
                 max_num_layers=max_num_layers, max_num_neurons=max_num_neurons,
                 evol_alg=evol_alg, sel=sel, lrate=lrate, cxp=cxp, mtp=mtp,
                 seed=seed, sel_kwargs=sel_kwargs, max_filter=max_filter, max_stride=max_stride,
                 hyperparameters=hyperparameters)
            print(x)
        except Exception as e:
            print('An error ocurred executing the {} dataset.'.format(dataset))    
            print(e)
            
        print('Time: ', time.time() - init_time)

def test_RNN(dataset_name, eval_func=None, batch_size=150, population=5, 
             generations=10, iters=100, max_num_layers=10, max_num_neurons=20,
             evol_alg='mu_plus_lambda', sel='best', lrate=0.01, cxp=0, mtp=1,
             seed=None, sel_kwargs={}, max_filter=4, max_stride=3,
             hyperparameters={}, is_time_series=True,
             series_input_width=30, series_label_width=1):
    """
    Tests the RNN network with the specified dataset and parameter selection.

    :param dataset_name: Name of the dataset that will be used in the genetic algorithm.
    :param eval_func: Evaluation function for evaluating each network.
    :param batch_size: Batch size of the data during the training of the networks.
    :param population: Number of individuals in the populations in the genetic algorithm.
    :param generations: Number of generations that will be done in the genetic algorithm.
    :param iters: Number of iterations that each network will be trained.
    :param max_num_layers: Maximum number of layers allowed in the networks.
    :param max_num_neurons: Maximum number of neurons allowed in the networks.
    :param max_filter: Maximum size of the filter allowed in the networks.
    :param max_stride: Maximum size of the stride allowed in the networks.
    :param evol_alg: Evolving algorithm that will be used during the genetic algorithm.
    :param sel: Selection method that will be used during the genetic algorithm.
    :param sel_kwargs: Arguments for selection method.
    :param lrate: Learning rate that will be used during training.
    :param cxp: Crossover probability that will be used during the genetic algorithm.
    :param mtp: Mutation probability that will be used during the genetic algorithm.
    :param seed: Seed that will be used in every random method.
    :param hyperparameters: Hyperparameters that will be evolved during the genetic algorithm.
    :param is_time_series: Boolean that indicates if the data is a time series.
    :param series_input_width: Width of the input series data. 
    :param series_label_width: Width of the labels series data. 
    :return: The last generation, a log book (stats) and the hall of fame (the best 
                 individuals found).
    """

    x_train, x_test, x_val, y_train, y_test, y_val, mode = load_dataset(dataset_name,is_time_series=is_time_series, 
                                        series_input_width=series_input_width, series_label_width=series_label_width)
    
    # Shape of logits must be 2
    if len(x_train.shape[1:]) == 3:
        x_train = x_train.reshape(x_train.shape[:-1])
        x_test = x_test.reshape(x_test.shape[:-1])
        x_val = x_val.reshape(x_val.shape[:-1])
    elif len(x_train.shape[1:]) == 1:
        x_train = x_train.reshape(list(x_train.shape) + [1])   
        x_test = x_test.reshape(list(x_test.shape) + [1])   
        x_val = x_val.reshape(list(x_val.shape) + [1])   

    input_shape = list(x_train.shape[1:])

    if len(y_train.shape) == 1:
        output_shape = [y_train.max() + 1]
    else:
        output_shape = y_train.shape[1:]
        
    if eval_func == None:
        eval_func = select_evaluation(mode)
    
    e = Evolving(evaluation=eval_func, 
			 desc_list=descriptors, 
			 x_trains=[x_train], y_trains=[y_train], 
			 x_tests=[x_val], y_tests=[y_val],
			 n_inputs=[input_shape],
			 n_outputs=[output_shape],
			 batch_size=batch_size,
			 population=population,
			 generations=generations,
			 iters=iters, 
             seed=seed,
             lrate=lrate,
             cxp=cxp,
             mtp=mtp,
             evol_alg=evol_alg,
             sel=sel,
             sel_kwargs=sel_kwargs,
			 max_num_layers=max_num_layers, 
			 max_num_neurons=max_num_neurons,
             max_filter=max_filter,
             max_stride=max_stride,
             hyperparameters=hyperparameters)   
     
    a = e.evolve()
    return a
        
def eval_rnn(nets, train_inputs, train_outputs, batch_size, iters, test_inputs, test_outputs, hypers):
    """
    The model is created with one RNN, but it needs a dense layer with a softmax activation function.
    That is needed because they are probability distributions and they have to be between 0 and 1.
    Finally accuracy error is used to measuare the performance of the model.
    
    :param nets: Dictionary with the networks that will be used to build the 
                 final network and that represent the individuals to be 
                 evaluated in the genetic algorithm.
    :param train_inputs: Input data for training, this data will only be used to 
                         give it to the created networks and train them.
    :param train_outputs: Output data for training, it will be used to compare 
                          the returned values by the networks and see their performance.
    :param batch_size: Number of samples per batch are used during training process.
    :param iters: Number of iterations that each network will be trained.
    :param test_inputs: Input data for testing, this data will only be used to 
                        give it to the created networks and test them. It can not be used during
                        training in order to get a real feedback.
    :param test_outputs: Output data for testing, it will be used to compare 
                         the returned values by the networks and see their real performance.
    :param hypers: Hyperparameters that are being evolved and used in the process.
    :return: Accuracy error obtained with the test data that evaluates the true
             performance of the network.
    """

    inp = Input(shape=train_inputs["i0"].shape[1:])
    out = nets["n0"].building(inp)
    if len(train_outputs["o0"].shape) == 1:
        out = Dense(10, activation='softmax')(out)
    else:    
        out = Dense(1)(out)
    
    model = Model(inputs=inp, outputs=out)
    #model.summary()
    opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])

    if len(train_outputs["o0"].shape) == 1:
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt, metrics=[])
    else:
        model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=opt, metrics=[])
        
    model.fit(train_inputs['i0'], train_outputs['o0'], epochs=iters, batch_size=batch_size, verbose=0)
                     
    preds = model.predict(test_inputs["i0"])
    
    if len(train_outputs["o0"].shape) == 1:
        res = tf.nn.softmax(preds)
        ev = accuracy_error(test_outputs["o0"], res)
    else:
        if np.isnan(preds).any():
            ev = 288
        else:
            ev = tf.keras.losses.mean_squared_error(preds, test_outputs["o0"])
            ev = np.sqrt(np.mean(ev))
        
    return ev, 
    
if __name__ == "__main__":
    #evaluated = test_RNN('estambul_values', is_time_series=True, series_input_width=50, series_label_width=1,
    #                     eval_func=eval_rnn, batch_size=300, population=4, 
    #                     generations=5, iters=3, max_num_layers=10, max_num_neurons=20,                  
    #                     hyperparameters = {"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]})
    test_RNN_all_datasets(is_time_series=True, series_input_width=50, series_label_width=1,
                          eval_func=eval_rnn, batch_size=150, population=2, 
                          generations=2, iters=10, max_num_layers=10, max_num_neurons=20,                  
                          hyperparameters = {"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]})