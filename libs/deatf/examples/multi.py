"""
This is a use case of DEATF where a mutiobjective problem is treated.

In order to face that problem, two MLPs are used. One will be responsible of
classifying the MNIST dataset and the other one of doing it with the fashion MNIST
dataset. Both are in the same model, but they work separately and do not interact
between them in any moment.
"""
import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np

from deatf.auxiliary_functions import load_fashion, load_mnist, accuracy_error
from deatf.network import MLPDescriptor
from deatf.evolution import Evolving

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model

def evaluation(nets, train_inputs, train_outputs, batch_size, iters, test_inputs, test_outputs, _):
    """
    In this caxe two simple MLPs are declarated and added to the model but 
    with different inputs and outputs, so they are separated. Both training and
    testing will be done with different data and two final results will be 
    obtained from the evaluation of the model.
    
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
    :return: Two accuracy errors, one from each MLP in the model.
    """
    
    inp_0 = Input(shape=train_inputs["i0"].shape[1:])
    out_0 = Flatten()(inp_0)
    out_0 = nets["n0"].building(out_0)
    
    inp_1 = Input(shape=train_inputs["i1"].shape[1:])
    out_1 = Flatten()(inp_1)
    out_1 = nets["n1"].building(out_1)
    
    model = Model(inputs=[inp_0, inp_1], outputs=[out_0, out_1])
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss=[tf.nn.softmax_cross_entropy_with_logits, 
                        tf.nn.softmax_cross_entropy_with_logits], 
                        optimizer=opt, metrics=[])
    
    # As the output has to be the same as the input, the input is passed twice
    model.fit([train_inputs['i0'], train_inputs['i1']],
              [train_outputs['o0'], train_outputs['o1']],
               epochs=iters, batch_size=batch_size, verbose=0)
    
    #tf.keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
        
    pred_0, pred_1 = model.predict([test_inputs['i0'], test_inputs['i1']])
        
    res_0 = tf.nn.softmax(pred_0)
    res_1 = tf.nn.softmax(pred_1)

    # Return both accuracies
    return accuracy_error(res_0, test_outputs["o0"]), accuracy_error(res_1, test_outputs["o1"])


if __name__ == "__main__":

    fashion_x_train, fashion_y_train, fashion_x_test, fashion_y_test, fashion_x_val, fashion_y_val = load_fashion()
    mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test, mnist_x_val, mnist_y_val = load_mnist()

    OHEnc = OneHotEncoder()

    fashion_y_train = OHEnc.fit_transform(np.reshape(fashion_y_train, (-1, 1))).toarray()
    fashion_y_test = OHEnc.fit_transform(np.reshape(fashion_y_test, (-1, 1))).toarray()
    fashion_y_val = OHEnc.fit_transform(np.reshape(fashion_y_val, (-1, 1))).toarray()

    mnist_y_train = OHEnc.fit_transform(np.reshape(mnist_y_train, (-1, 1))).toarray()
    mnist_y_test = OHEnc.fit_transform(np.reshape(mnist_y_test, (-1, 1))).toarray()
    mnist_y_val = OHEnc.fit_transform(np.reshape(mnist_y_val, (-1, 1))).toarray()

    # In this case, we provide two data inputs and outputs
    e = Evolving(evaluation=evaluation, desc_list=[MLPDescriptor, MLPDescriptor],
                 x_trains=[fashion_x_train, mnist_x_train], y_trains=[fashion_y_train, mnist_y_train], 
                 x_tests=[fashion_x_val, mnist_x_val], y_tests=[fashion_y_val, mnist_y_val], 
                 n_inputs=[[28, 28], [28, 28]], n_outputs=[[10], [10]],
                 population=5, generations=5, batch_size=150, iters=50, 
                 lrate=0.1, cxp=0, mtp=1, seed=0,
                 max_num_layers=10, max_num_neurons=100, max_filter=4, max_stride=3,
                 evol_alg='mu_plus_lambda', sel='best', sel_kwargs={}, 
                 hyperparameters={}, 
                 batch_norm=True, dropout=True)

    res = e.evolve()

    print(res[0])
