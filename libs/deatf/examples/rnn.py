"""
This is a use case of DEATF where a RNN is used.

This is a classification problem with the fasion MNIST dataset. In this case 
a RNN is used to procces that data and those images are passed as sequences
and with those the RNN has to return and predict the class of each image.
"""
import sys
sys.path.append('..')

import tensorflow as tf

from deatf.auxiliary_functions import accuracy_error, load_fashion
from deatf.network import RNNDescriptor
from deatf.evolution import Evolving

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as opt


optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]


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
    out = Dense(10, activation='softmax')(out) # As they are probability distributions, they have to be bewteen 0 an 1
    model = Model(inputs=inp, outputs=out)
    
    model.summary()
    
    opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt, metrics=[])
    
    model.fit(train_inputs['i0'], train_outputs['o0'], epochs=iters, batch_size=batch_size, verbose=0)
                     
    preds = model.predict(test_inputs["i0"])
    
    res = tf.nn.softmax(preds)

    return accuracy_error(test_outputs["o0"], res),

if __name__ == "__main__":

    x_train, y_train, x_test, y_test, x_val, y_val = load_fashion()

    # Normalize the input data
    x_train = x_train/255
    x_test = x_test/255
    x_val = x_val/255
    
    # Here we define a convolutional-transposed convolutional network combination
    e = Evolving(evaluation=eval_rnn, desc_list=[RNNDescriptor], 
                 x_trains=[x_train], y_trains=[y_train], x_tests=[x_val], y_tests=[y_val], 
                 n_inputs=[[28, 28]], n_outputs=[[10]], 
                 population=5, generations=5, batch_size=150, iters=50, 
                 lrate=0.1, cxp=0, mtp=1, seed=0,
                 max_num_layers=10, max_num_neurons=100, max_filter=4, max_stride=3,
                 evol_alg='mu_plus_lambda', sel='best', sel_kwargs={}, 
                 hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]}, 
                 batch_norm=True, dropout=True)    

    a = e.evolve()

    print(a[-1])
