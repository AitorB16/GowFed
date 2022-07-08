"""
This is a use case of DEATF where a sequential combination of two networks is used.

The two networks that are combined in a sequential way are two MLPs.
This is a classification problem with fashion MNIST dataset. Due to
the two dimensions of the input data, it has to be flattened in order
to pass it to the MLPs; but the rest is similar to other classification 
examples in this library.
"""
import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np

from deatf.auxiliary_functions import accuracy_error, load_fashion
from deatf.network import MLPDescriptor
from deatf.evolution import Evolving

from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as opt
from sklearn.preprocessing import OneHotEncoder

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]


def eval_sequential(nets, train_inputs, train_outputs, batch_size, iters, test_inputs, test_outputs, hypers):
    """
    The model that will evaluate the data is formed by a MLP and then other MLP.
    As they are compatible between them, there is no need of extra layers between 
    them. Softmax cross entropy is used for the training and accuracy error for 
    evaluating the network.    
    
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
    out = Flatten()(inp)
    out = nets["n0"].building(out)
    out = nets["n1"].building(out)

    model = Model(inputs=inp, outputs=out)
    
    opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])
    
    model.compile(loss=tf.nn.softmax_cross_entropy_with_logits, optimizer=opt, metrics=[])
    
    model.fit(train_inputs['i0'], train_outputs['o0'], epochs=iters, batch_size=batch_size, verbose=0)

    pred = model.predict(test_inputs['i0'])
        
    res = tf.nn.softmax(pred)

    return accuracy_error(test_outputs["o0"], res),


if __name__ == "__main__":

    x_train, y_train, x_test, y_test, x_val, y_val = load_fashion()

    OHEnc = OneHotEncoder()

    y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()
    y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
    y_val = OHEnc.fit_transform(np.reshape(y_val, (-1, 1))).toarray()
    
    # When calling the function, we indicate the training function, what we want to evolve (two MLPs), input and output data for training and
    # testing, fitness function, batch size, population size, number of generations, input and output dimensions of the networks, crossover and
    # mutation probability, the hyperparameters being evolved (name and possibilities), and whether batch normalization and dropout should be
    # present in evolution

    e = Evolving(evaluation=eval_sequential, desc_list=[MLPDescriptor, MLPDescriptor], 
                 x_trains=[x_train], y_trains=[y_train], x_tests=[x_val], y_tests=[y_val], 
                 n_inputs=[[28, 28], [10]], n_outputs=[[10], [10]],
                 population=5, generations=5, batch_size=150, iters=50, 
                 lrate=0.1, cxp=0.5, mtp=0.5, seed=0,
                 max_num_layers=10, max_num_neurons=100, max_filter=4, max_stride=3,
                 evol_alg='mu_plus_lambda', sel='best', sel_kwargs={}, 
                 hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]}, 
                 batch_norm=False, dropout=False)

    a = e.evolve()
    print(a[-1])
