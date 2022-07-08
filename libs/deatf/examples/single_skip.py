"""
This is a use case of DEATF where a skip CNN is used.

This is a classification problem with fashion MNIST dataset. This example is similar
to the cnn_class.py file; but, in this case the CNN used has skips in its structure.
Those skips are conections that are made from a layer with layers that are after it.
"""
import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np

from deatf.auxiliary_functions import accuracy_error, load_fashion
from deatf.network import MLPDescriptor, CNNDescriptor, CNN
from deatf import evolution

from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Concatenate, UpSampling2D
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as opt
from sklearn.preprocessing import OneHotEncoder


class SkipCNN(CNN):
    """
    This network inherits from CNN that in turn inherits from Network (what 
    SkipCNN also does). The parameters are the same as the CNN, it only 
    rewrites the building function. In it a new parameter is added,
    that is the skip added to the network from the beginig to the selected
    layer, that is the difference.
    
    :param network_descriptor: Descriptor of the CNN.
    """
    def building(self, x, skip):
        """
        Given a TensorFlow layer, this functions continues adding more layers of a SkipCNN.
        
        :param x: A layer from TensorFlow.
        :param skip: Number of the layer it has to do the skip into. If the number is
                     greater than the number of layers, it will be calculated and reasigned a new value.
        :return: The layer received from parameter with the SkipCNN concatenated to it.
        """       
        
        skip = (self.descriptor.number_hidden_layers % (skip-2)) + 2
        for lay_indx in range(self.descriptor.number_hidden_layers):
            
            if lay_indx == 0:
                skip_layer = x
                skip_kernel_size = x.shape[1]
            
            if skip == lay_indx:
                actual_kernel_size = x.shape[1]
                x = UpSampling2D((skip_kernel_size,skip_kernel_size))(x)
                x = MaxPooling2D((actual_kernel_size, actual_kernel_size))(x)
                x = Concatenate(axis=-1)([x, skip_layer])
                
            if self.descriptor.layers[lay_indx] == 2:  # If the layer is convolutional
                
                x = Conv2D(self.descriptor.filters[lay_indx][2],
                           [self.descriptor.filters[lay_indx][0],self.descriptor.filters[lay_indx][1]],
                           strides=[self.descriptor.strides[lay_indx][0], self.descriptor.strides[lay_indx][1]],
                           padding="valid",
                           activation=self.descriptor.act_functions[lay_indx],
                           kernel_initializer=self.descriptor.init_functions[lay_indx])(x)

            elif self.descriptor.layers[lay_indx] == 0:  # If the layer is average pooling
                x = AveragePooling2D(pool_size=[self.descriptor.filters[lay_indx][0], self.descriptor.filters[lay_indx][1]],
                                           strides=[self.descriptor.strides[lay_indx][0], self.descriptor.strides[lay_indx][1]],
                                           padding="valid")(x)
            else:
                x = MaxPooling2D(pool_size=[self.descriptor.filters[lay_indx][0], self.descriptor.filters[lay_indx][1]],
                                       strides=[self.descriptor.strides[lay_indx][0], self.descriptor.strides[lay_indx][1]],
                                       padding="valid")(x)
                
        return x

evolution.descs["CNNDescriptor"] = SkipCNN

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]

def eval_cnn(nets, train_inputs, train_outputs, batch_size, iters, test_inputs, test_outputs, hypers):
    """
    Creates the model formed by the SkipCNN that has been created and then
    a MLP is sequenetialy added with a flatten and a dense layer in between.
    That model is trained by using cross entropy function and the final 
    evaluation is done with the accuracy error.
    
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
    out = nets["n0"].building(inp, hypers["skip"])
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
    
    x_train = x_train[:500]
    y_train = y_train[:500]
    x_test = x_test[:100]
    y_test = y_test[:100]
    x_val = x_val[:100]
    y_val = y_val[:100]
    
    
    # We fake a 3 channel dataset by copying the grayscale channel three times.
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

    # Here we indicate that we want a CNN as the first network of the model
    e = evolution.Evolving(evaluation=eval_cnn, desc_list=[CNNDescriptor, MLPDescriptor], 
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
