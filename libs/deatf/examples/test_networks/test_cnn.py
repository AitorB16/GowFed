import sys
sys.path.append('.')
 
import tensorflow as tf
import time

from deatf.auxiliary_functions import accuracy_error
from deatf.network import CNNDescriptor

from aux_functions_testing import test

import tensorflow.keras.optimizers as opt
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]


def test_CNN_all_datasets(eval_func=None, batch_size=150, population=5, 
                      generations=10, iters=100, max_num_layers=10, max_num_neurons=20,
                      evol_alg='mu_plus_lambda', sel='best', lrate=0.01, cxp=0, mtp=1,
                      seed=None, sel_kwargs={}, max_filter=4, max_stride=3):
    """
    Tests the CNN network with all the possible datasets and with the specified parameter selection.

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
    """

    dataset_collection = ['mnist', 'kmnist', 'cmaterdb', 'fashion_mnist', 
                          'binary_alpha_digits', 'cifar10', 'rock_paper_scissors']

    for dataset in dataset_collection:
        
        print('\nEvaluating the {} dataset with the following configuration:'.format(dataset),
              '\nBatch size:  {}\nPopulation of networks:  {}\nGenerations:  {}'.format(batch_size, population, generations),
              '\nIterations in each network:  {}\nMaximum number of layers:  {}'.format(iters, max_num_layers),
              '\nMaximum number of neurons in each layer: {}'.format(max_num_neurons))

        init_time = time.time()
        
        try:
            x = test_CNN(dataset, eval_func=eval_func, batch_size=batch_size, 
                 population=population, generations=generations, iters=iters, 
                 max_num_layers=max_num_layers, max_num_neurons=max_num_neurons,
                 evol_alg=evol_alg, sel=sel, lrate=lrate, cxp=cxp, mtp=mtp,
                 seed=seed, sel_kwargs=sel_kwargs, max_filter=max_filter, max_stride=max_stride)
            print(x)
        except Exception as e:
            print('An error ocurred executing the {} dataset.'.format(dataset))    
            print(e)
            
        print('Time: ', time.time() - init_time)
        
def test_CNN(dataset_name, eval_func=None, batch_size=150, population=5, 
             generations=10, iters=100, max_num_layers=10, max_num_neurons=20,
             evol_alg='mu_plus_lambda', sel='best', lrate=0.01, cxp=0, mtp=1,
             seed=None, sel_kwargs={}, max_filter=4, max_stride=3):
    """
    Tests the CNN network with the specified dataset and parameter selection.

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
    :return: The last generation, a log book (stats) and the hall of fame (the best 
                 individuals found).
    """

    return test(dataset_name, descriptors=[CNNDescriptor], eval_func=eval_func, 
                batch_size=batch_size, population=population, generations=generations, 
                iters=iters, max_num_layers=max_num_layers, max_num_neurons=max_num_neurons,  
                hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]},
                evol_alg=evol_alg, sel=sel, lrate=lrate, cxp=cxp, mtp=mtp,
                seed=seed, sel_kwargs=sel_kwargs, max_filter=max_filter, max_stride=max_stride)
        
def eval_cnn(nets, train_inputs, train_outputs, batch_size, iters, test_inputs, test_outputs, hypers):
    """
    Evaluation method for the CNN. Softmax cross entropy is used for the 
    training and accuracy error for evaluating the network.

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
    out = Flatten()(out)
    out = Dense(train_outputs["o0"].shape[1])(out)
    
    model = Model(inputs=inp, outputs=out)

    opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])
    model.compile(loss=tf.nn.softmax_cross_entropy_with_logits, optimizer=opt, metrics=[])
    
    model.fit(train_inputs['i0'], train_outputs['o0'], epochs=iters, batch_size=batch_size, verbose=0)

    preds = model.predict(test_inputs["i0"])
    
    res = tf.nn.softmax(preds)

    return accuracy_error(test_outputs["o0"], res),
    
if __name__ == "__main__":
    #evaluated = test_CNN('binary_alpha_digits', eval_func=eval_cnn, batch_size=150, population=20, generations=5, iters=10)
    test_CNN_all_datasets(eval_func=eval_cnn, batch_size=150, population=2, generations=2, iters=5)