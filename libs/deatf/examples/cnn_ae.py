"""
This is a use case of DEATF where a convolutional Autoencoder is used.

In order to create that Autoencoder both CNN and TCNN are used. This is an unsupervised
problem, where the objective is reducing the dimensionaly of the data. The CNN wil be 
responisble of reducing the dimension and the TCNN of returning it into its original form.
"""
import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder


from deatf.network import CNNDescriptor, TCNNDescriptor
from deatf.auxiliary_functions import load_fashion
from deatf.evolution import Evolving

from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as opt

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]

def eval_cnn_ae(nets, train_inputs, _, batch_size, iters, test_inputs, __, hypers):
    """
    In order to evaluate the Autoencoder, as an MLP descriptor is used,
    a Flatten layer is added before the network that is created.
    Then is trained using the defined mean square error and its 
    final performance metric is also the mean squared error.
    
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
    :return: Mean squared error obtained with the test data that evaluates the true
             performance of the network.
    """
  
    inp = Input(shape=train_inputs["i0"].shape[1:])
    out = nets["n0"].building(inp)
    out = nets["n1"].building(out)
    
    model = Model(inputs=inp, outputs=out)

    opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])
    model.compile(loss=tf.losses.mean_squared_error, optimizer=opt, metrics=[])
    
    # As the output has to be the same as the input, the input is passed twice
    model.fit(train_inputs['i0'], train_inputs['i0'], epochs=iters , batch_size=batch_size, verbose=0)
                
    pred = model.predict(test_inputs["i0"])
    res = pred[:, :28, :28, :3] 
        
    if np.isnan(res).any():
        return 288,
    else:
        return mean_squared_error(np.reshape(res, (-1)), np.reshape(test_inputs["i0"], (-1))),


if __name__ == "__main__":

    x_train, _, x_test, _, x_val, _ = load_fashion()
        
    x_train = x_train[:10000]
    x_test = x_test[:5000]
    x_val = x_val[:5000]
    
    # 3 channel dataset is faked by copying the grayscale channel three times.
    x_train = np.expand_dims(x_train, axis=3)/255
    x_train = np.concatenate((x_train, x_train, x_train), axis=3)

    x_test = np.expand_dims(x_test, axis=3)/255
    x_test = np.concatenate((x_test, x_test, x_test), axis=3)

    x_val = np.expand_dims(x_val, axis=3)/255
    x_val = np.concatenate((x_val, x_val, x_val), axis=3)

    e = Evolving(evaluation=eval_cnn_ae, desc_list=[CNNDescriptor, TCNNDescriptor], 
                 x_trains=[x_train], y_trains=[x_train], x_tests=[x_val], y_tests=[x_val], 
                 n_inputs=[[28, 28, 3], [7, 7, 1]], n_outputs=[[7,7,1], [28, 28, 3]],
                 population=5, generations=5, batch_size=150, iters=50, 
                 lrate=0.1, cxp=0.5, mtp=0.5, seed=0,
                 max_num_layers=10, max_num_neurons=100, max_filter=4, max_stride=3,
                 evol_alg='mu_plus_lambda', sel='best', sel_kwargs={}, 
                 hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]}, 
                 batch_norm=True, dropout=True)

    a = e.evolve()
    
    print(a[-1])
    
