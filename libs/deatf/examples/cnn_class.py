"""
This is a use case of DEATF where a CNN and a MLP are combined with 
that order in a sequential way.

This is a classification problem with fashion MNIST dataset. As the first
component of the network is a CNN, the input data (clothes' images) can be 
directly passed and the output of the MLP is the result of the classification.
"""
import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np

from deatf.auxiliary_functions import accuracy_error, load_fashion
from deatf.network import MLPDescriptor, CNNDescriptor
from deatf.evolution import Evolving

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as opt

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]

def eval_cnn(nets, train_inputs, train_outputs, batch_size, iters, test_inputs, test_outputs, hypers):
    """
    The model that will evaluate the data is formed by a CNN and then the MLP.
    Between those subnetwork, a flatten layer and a dense layer are added in order
    to enable the union. Softmax cross entropy is used for the training and accuracy error for 
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
    out = nets["n0"].building(inp)
    out = Flatten()(out)
    out = Dense(20)(out)
    out = nets["n1"].building(out)
    
    model = Model(inputs=inp, outputs=out)

    opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])
    model.compile(loss=tf.nn.softmax_cross_entropy_with_logits, optimizer=opt, metrics=[])
    
    model.fit(train_inputs['i0'], train_outputs['o0'], epochs=iters, batch_size=batch_size, verbose=0)
    
    preds = model.predict(test_inputs["i0"])
    
    res = tf.nn.softmax(preds)

    return accuracy_error(test_outputs["o0"], res),

if __name__ == "__main__":

    x_train, y_train, x_test, y_test, x_val, y_val = load_fashion()
    
    x_train = x_train[:10000]
    y_train = y_train[:10000]
    x_test = x_test[:5000]
    y_test = y_test[:5000]
    x_val = x_val[:5000]
    y_val = y_val[:5000]
    
    # 3 channel dataset is faked by copying the grayscale channel three times.
    x_train = np.expand_dims(x_train, axis=3)/255
    x_train = np.concatenate((x_train, x_train, x_train), axis=3)

    x_test = np.expand_dims(x_test, axis=3)/255
    x_test = np.concatenate((x_test, x_test, x_test), axis=3)

    x_val = np.expand_dims(x_val, axis=3)/255
    x_val = np.concatenate((x_val, x_val, x_val), axis=3)
    
    OHEnc = OneHotEncoder()

    y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()

    y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
    
    y_val = OHEnc.fit_transform(np.reshape(y_val, (-1, 1))).toarray()  
    
    # The order in the descriptor list is important in order to have the CNN first
    e = Evolving(evaluation=eval_cnn, desc_list=[CNNDescriptor, MLPDescriptor], 
                 x_trains=[x_train], y_trains=[y_train], x_tests=[x_val], y_tests=[y_val], 
                 n_inputs=[[28, 28, 3], [20]], n_outputs=[[7, 7, 1], [10]], 
                 population=5, generations=5, batch_size=150, iters=50, 
                 lrate=0.1, cxp=0.5, mtp=0.5, seed=0,
                 max_num_layers=10, max_num_neurons=100, max_filter=4, max_stride=3,
                 evol_alg='mu_plus_lambda', sel='best', sel_kwargs={}, 
                 hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]}, 
                 batch_norm=True, dropout=True)

    a = e.evolve()

    print(a[-1])
