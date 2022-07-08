"""
This is a use case of DEATF where a weighted artificial neural network (WANN) is used.

This is a case like the simple.py example because only one MLP is used to solve the 
problem. The difference is that here the weights are not randomly initialized, they
are loaded to the model like hyperparameters and as hyperparameters they are evolved.
So is an aproach of weight evolving. This is a classification problem, but in order
to simplify it it has be turned into a binary classification problem.
"""
import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np

from deatf.network import MLPDescriptor
from deatf.evolution import Evolving
from deatf.auxiliary_functions import load_fashion

from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as opt
from sklearn.preprocessing import OneHotEncoder

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]


def eval_wann(nets, train_inputs, train_outputs, batch_size, iters, test_inputs, test_outputs, hypers):
    """
    This evaluation case is different to the ones in the other examples and 
    not because of the model's building. The difference is that the weigths 
    are loaded into the model and this implies that it do not need to train,
    once the weights are loaded the model can predict. In orddr to evaluate 
    the performance it is used the mean of the cases that have been well 
    predicted.
    
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
    :return: Mean of the cases that have been well predicted that evaluates the true
             performance of the network.
    """

    inp = Input(shape=train_inputs["i0"].shape[1:])
    out = Flatten()(inp)
    out = nets["n0"].building(out)
    
    model = Model(inputs=inp, outputs=out)
    
    parameters = 0
    for layer in model.layers:
        if layer.get_weights() != []:
            parameters += np.prod(layer.get_weights()[0].shape)
    
    string = hypers["start"]
    while len(string) < parameters:
        string = string.replace("0", hypers["p1"])
        string = string.replace("1", hypers["p2"])
    aux = 0
    ls = []
    for layer in model.layers:
        lay = []
        if layer.get_weights() != []:
            if len(layer.get_weights()[0].shape) == 2:
                for i in range(layer.get_weights()[0].shape[0]):
                    lay += [[int(i) for i in string[aux:aux+layer.get_weights()[0].shape[1]]]]
                    aux += layer.get_weights()[0].shape[1]
                lay = np.array(lay)
                lay = np.where(lay == 0, hypers["weight1"], lay)
                lay = np.where(lay == 1, hypers["weight2"], lay)
                lay = [lay, layer.get_weights()[1]]
        ls += [lay]
    
    for i, layer in enumerate(model.layers):  # Here, the weights are assigned
        if ls[i]:
            layer.set_weights(ls[i])
    
    preds = model.predict(test_inputs["i0"])
    
    res = tf.nn.softmax(preds)
    
    res = np.argmax(res, axis=1)
    res = 1 - np.sum(np.argmax(test_outputs["o0"], axis=1) == res) / res.shape[0]

    return res,


if __name__ == "__main__":

    x_train, y_train, x_test, y_test, x_val, y_val = load_fashion()
    
    x_train = x_train/255
    x_test = x_test/255
    x_val = x_val/255
    
    y_train = np.array([0 if x <= 4 else 1 for x in y_train])
    y_test = np.array([0 if x <= 4 else 1 for x in y_test])
    y_val = np.array([0 if x <= 4 else 1 for x in y_val])

    OHEnc = OneHotEncoder()

    y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()
    y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
    y_val = OHEnc.fit_transform(np.reshape(y_val, (-1, 1))).toarray()


    e = Evolving(evaluation=eval_wann, desc_list=[MLPDescriptor], 
                 x_trains=[x_train], y_trains=[y_train], x_tests=[x_val], y_tests=[y_val],
                 n_inputs=[[28, 28]], n_outputs=[[2]],
                 population=5, generations=5, batch_size=150, iters=50, 
                 lrate=0.1, cxp=0, mtp=1, seed=0,
                 max_num_layers=10, max_num_neurons=100, max_filter=4, max_stride=3,
                 evol_alg='mu_plus_lambda', sel='best', sel_kwargs={}, 
                 hyperparameters={"weight1": np.arange(-2, 2, 0.5), "weight2": np.arange(-2, 2, 0.5), 
                                  "start": ["0", "1"], "p1": ["01", "10"],
                                  "p2": ["001", "010", "011", "101", "110", "100"]}, 
                 batch_norm=False, dropout=False) # The weights, that are also evolved

    a = e.evolve()
    print(a[-1])
