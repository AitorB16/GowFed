"""
This is a use case of DEATF where an Autoencoder is used. 

In order to create the Autoencoder model, two MLP networks are used. This is an unsupervised
problem, where the objective is reducing the dimensionaly of the data. One MLP wil be 
responisble of reducing the dimension and the other MLP of returning it into its original form.
"""

import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np


from deatf.auxiliary_functions import load_fashion
from deatf.network import MLPDescriptor
from deatf.evolution import Evolving

from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model

def hand_made_tf_mse(target, prediction):
    """
    Calculates the mean squared error by using TensorFlow operators.
    
    :param target: Ground of truth data.
    :param prediction: Predicted data.
    :return: Mean squared error in a Tensor form.
    """
    return tf.reduce_mean(tf.math.squared_difference(target, prediction))

def hand_made_np_mse(target, prediction):
    """
    Calculates the mean squared error by using NumPy operators.
    
    :param target: Ground of truth data.
    :param prediction: Predicted data.
    :return: Mean squared error in a NumPy form.
    """
    return np.mean(np.square(target-prediction))

def ae_eval(nets, train_inputs, train_outputs, batch_size, iters, test_inputs, test_outputs, hypers):
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
    out = Flatten()(inp)
    out = nets['n0'].building(out)
    model = Model(inputs=inp, outputs=out)

    opt = tf.keras.optimizers.Adam(learning_rate=hypers["lrate"])
    model.compile(loss=hand_made_tf_mse, optimizer=opt, metrics=[])
    
    model.fit(train_inputs['i0'], train_inputs['i0'], epochs=iters, batch_size=batch_size, verbose=0)
    
    prediction = model.predict(test_inputs['i0'])
    
    ev = hand_made_np_mse(test_inputs['i0'], prediction)
    
    if isinstance(ev, float):
        ev = (ev,)
        
    return ev

if __name__ == "__main__":
    
    x_train, _, x_test, _, x_val, _ = load_fashion()

    x_train = np.reshape(x_train, (-1, 784))
    x_test = np.reshape(x_test, (-1, 784))
    x_val = np.reshape(x_val, (-1, 784))

    x_train = x_train/1
    x_test = x_test/1
    x_val = x_val/1


    e = Evolving(evaluation=ae_eval, desc_list=[MLPDescriptor], 
                 x_trains=[x_train], y_trains=[x_train], x_tests=[x_val], y_tests=[x_val], 
                 n_inputs=[[784]], n_outputs=[[784]],
                 population=5, generations=5, batch_size=150, iters=50, 
                 lrate=0.1, cxp=0, mtp=1, seed=0,
                 max_num_layers=10, max_num_neurons=100, max_filter=4, max_stride=3,
                 evol_alg='mu_plus_lambda', sel='best', sel_kwargs={}, 
                 hyperparameters={"lrate": [0.1, 0.5, 1]}, 
                 batch_norm=True, dropout=True)    

    a = e.evolve()

    print(a)
