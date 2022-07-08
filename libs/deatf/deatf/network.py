"""
Here there can be found two relevant classes in the library: :class:`~NetworkDescriptor` and 
:class:`~Network`. These classes are directly linked between them and both have a principal
and more abstract class and then one class for each network type that inherit from them.

:class:`~NetworkDescriptor` and its subclasses are the ones responsible of establishing
the structure and atributes that the network will have. All these classes have functions
for a random initialization (to initialize individuals in the genetic algorithm) and
functions to make changes and make easier the mutations. This would be the genotype in the 
evolutionary algorithm.

:class:`~NetworkDescriptor` and its subclasses are otherwise the phenotype, they are 
the network itself constructed by using the TensorFlow library. That is why they only
have one function, to turn the received descriptor into a network that contains all 
that information.

========================================================================================================
"""

import tensorflow as tf
import numpy as np
import os
import copy

from functools import reduce

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, \
                                    MaxPooling2D, AveragePooling2D, Bidirectional, SimpleRNN, LSTM, GRU
from tensorflow.keras.initializers import RandomNormal, RandomUniform, GlorotNormal, GlorotUniform

activations = [None, tf.nn.relu, tf.nn.elu, tf.nn.softplus, tf.nn.softsign, tf.sigmoid, tf.nn.tanh]
initializations = [RandomNormal, RandomUniform, GlorotNormal, GlorotUniform]

MIN_NUM_NEURONS = 2
MIN_NUM_FILTERS = 1
MIN_NUM_STRIDES = 1
MIN_NUM_CHANNELS = 3
MAX_NUM_CHANNELS = 65

class NetworkDescriptor:
    """
    This class implements the descriptor of a generic network. Subclasses of this are the evolved ones.
    Parameters described in this class are the ones that are in all the types of networks
    
    :param number_hidden_layers: Number of hidden layers in the network.
    :param input_dim: Dimension of the input data (can have one or more dimensions).
    :param output_dim: Expected output of the network (similarly, can have one or more dimensions).
    :param init_functions: Weight initialization functions. Can have different ranges, depending on the subclass.
    :param act_functions: Activation functions to be applied after each layer.
    :param batch_norm: A boolean that indicates if batch normalization is applied after each layer in the network.
    :param dropout: A boolean that indicates if dropout is applied after each layer in the network.
    :param dropout_probs: The different probabilities of the dropout after each layer.
    """
    def __init__(self, number_hidden_layers=None, input_dim=None, output_dim=None, 
                 max_num_layers=None, max_num_neurons=None,
                 init_functions=[], act_functions=[], 
                 dropout=False, dropout_probs=[], batch_norm=False):
        self.number_hidden_layers = number_hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init_functions = init_functions
        self.act_functions = act_functions
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.dropout_probs = dropout_probs
        
        self.max_num_layers = max_num_layers
        self.max_num_neurons = max_num_neurons

    def remove_layer(self, layer_pos):  
        """
        Removes the layer in the position received by parameter from the network.
        
        :param layer_pos: Position of the layer that is going to be removed.
        """
        # Defined just in case the user redefines classes and forgets to define this function
        pass

    def remove_random_layer(self):
        """
        Selects and removes a random layer from the network.
        
        :return: True if the action is done or False if it can not be done.
        """
        layer_pos = np.random.randint(self.number_hidden_layers)
        return self.remove_layer(layer_pos)

    def change_activation(self, layer_pos, new_act_fn):
        """
        Exchanges the activation function in the selected layer for a new one passed by parameter.
        
        :param layer_pos: Position of the layer whose activation function wants to be changed.
        :param new_act_fn: New activation fucntion that is going to be asigned.
        :return: True if the action is done or False if it can not be done (in this case
                 it will always be done).
        """
        self.act_functions[layer_pos] = new_act_fn
        return True

    def change_weight_init(self, layer_pos, new_weight_fn):
        """
        Exchanges the weight initialization function in the selected layer for a new one passed by parameter.
        
        :param layer_pos: Position of the layer whose activation function wants to be changed.
        :param new_act_fn: New activation fucntion that is going to be asigned.
        :return: True if the action is done or False if it can not be done (in this case
                 it will always be done).
        """
        self.init_functions[layer_pos] = new_weight_fn
        return True

    def change_dropout(self):
        """
        Change the dropout conditional. If dropout layers are used, quit them; otherwise, add them.
        
        :return: True if the action is done or False if it can not be done (in this case
                 it will always be done).
        """
        self.dropout = not self.dropout
        return True # This mutation is controled, it always will be applied
    
    def change_dropout_prob(self):
        """
        The dropout probability for each layer is changed by a new random one (between 0 and 1).
        
        :return: True if the action is done or False if it can not be done (in this case
                 it will always be done).
        """
        self.dropout_probs = np.random.rand(self.number_hidden_layers)
        return True # This mutation is controled, it always will be applied
    
    def change_batch_norm(self):
        """
        Change the batch normalization conditional. If batch normalization is used, quit it; otherwise, add it.
        
        :return: True if the action is done or False if it can not be done (in this case
                 it will always be done).
        """
        self.batch_norm = not self.batch_norm
        return True # This mutation is controled, it always will be applied
        
    def print_components(self):
        """
        Codifies all the components of the network in a string.
        
        :return: A string with all the components of the network.
        """
        components = self.codify_components()
        
        text = f"{self.__name__()} has the following parameters: \n"
        for component in components:
            if isinstance(component[1], list) or isinstance(component[1], np.ndarray):
                text += f"\t-{component[0]}:\n"
                for lay_indx, element in enumerate(component[1]):
                    try:
                        text += f"\t\tLayer {lay_indx} -> {element.__name__}\n"
                    except AttributeError:
                        text += f"\t\tLayer {lay_indx} -> {element}\n"
            else:
                text += f"\t-{component[0]}: {component[1]}\n"

        return text


    def __str__(self):
        return self.print_components()

class MLPDescriptor(NetworkDescriptor):
    """
    Descriptor of a Multi Layer Perceptron, formed by hidden layers and with the possibility of
    selecting the number of neurons in each layer and including dropout and batch normalization.
    
    :param number_hidden_layers: Number of hidden layers in the network.
    :param input_dim: Dimension of the input data (can have one or more dimensions).
    :param output_dim: Expected output of the network (similarly, can have one or more dimensions).
    :param init_functions: Weight initialization functions. Can have different ranges, depending on the subclass.
    :param act_functions: Activation functions to be applied after each layer.
    :param batch_norm: A boolean that indicates if batch normalization is applied after each layer in the network.
    :param dropout: A boolean that indicates if dropout is applied after each layer in the network.
    :param dropout_probs: The different probabilities of the dropout after each layer.

    :param dims: Number of neurons in each layer.
    """
    def __init__(self, number_hidden_layers=None, input_dim=None, output_dim=None,  
                 dims=[], init_functions=[], act_functions=[], 
                 dropout=False, dropout_probs=[], batch_norm=False):
        
        super().__init__(number_hidden_layers=number_hidden_layers, input_dim=input_dim, output_dim=output_dim, 
                         init_functions=init_functions, act_functions=act_functions, 
                         dropout=dropout, dropout_probs=dropout_probs, batch_norm=batch_norm)
        self.dims = dims  # Number of neurons in each layer

    def random_init(self, input_size, output_size, max_num_layers, max_num_neurons, max_stride, max_filter, dropout, batch_norm):
        """
        Given the input, the output and some limits for the parametes of the 
        network, a random initialization of the object (the network descriptor) is done. 
        
        :param input_size: Input size of the network (it will be flattened in order to fit in the MLP).
        :param output_size: Output size of the network (it will be flattened in order to fit in the MLP).
        :param max_num_layers: Maximum number of layers that can be in the network.
        :param max_num_neurons: Maximum number fo units that can be in each recurrent layer of the network.
        :param max_stride: Maximum stride possible (used as 2).
        :param max_filter: Maximum filter size possible (used as 3).
        :param dropout: A boolean value that indicates if the networks can have dropout.
        :param batch_norm: A boolean value that indicates if the networks can have batch normalization.
        """
        
        self.max_num_layers = max_num_layers
        self.max_num_neurons = max_num_neurons
        
        if hasattr(input_size, '__iter__'):
            # If the incoming/outgoing sizes have more than one dimension compute the size of the flattened sizes
            self.input_dim = reduce(lambda x, y: x*y, input_size)
        else:
            self.input_dim = input_size
            
        if hasattr(output_size, '__iter__'):
            self.output_dim = reduce(lambda x, y: x*y, output_size)
        else:
            self.output_dim = output_size

        self.number_hidden_layers = np.random.randint(max_num_layers)+1
        
        self.dims = [np.random.randint(MIN_NUM_NEURONS, max_num_neurons) for _ in range(self.number_hidden_layers)]
        self.init_functions = np.random.choice(initializations, size=self.number_hidden_layers)
        self.act_functions = np.random.choice(activations, size=self.number_hidden_layers)
        
        if batch_norm:
            self.batch_norm = np.random.choice([True, False])
            
        if dropout:
            self.dropout = np.random.choice([True, False])
            self.dropout_probs = np.random.rand(self.number_hidden_layers)
        else:
            self.dropout_probs = np.zeros(self.number_hidden_layers)

    def add_layer(self, layer_pos, lay_dims, init_function, act_function, drop_prob):
        """
        Adds a layer in the given position with all the characteristics indicated by parameters.
        
        :param layer_pos: Position of the layer.
        :param lay_dims: Number of neurons in the layer.
        :param init_function: Function for initializing the weights of the layer.
        :param act_function: Activation function to be applied.
        :param drop_prob: Probability of dropout.
        :param batch_norm: Whether batch normalization is applied after the layer.
        :return: True if the action is done or False if it can not be done.
        """
        if self.number_hidden_layers >= self.max_num_layers:
            return False 
        if lay_dims < MIN_NUM_NEURONS or lay_dims > self.max_num_neurons:
            return False # It is not within feasible bounds
        
        # We create the new layer and add it to the network descriptor
        self.dims = np.insert(self.dims, layer_pos, lay_dims)
        self.init_functions = np.insert(self.init_functions, layer_pos, init_function)
        self.act_functions = np.insert(self.act_functions, layer_pos, act_function)
        self.number_hidden_layers = self.number_hidden_layers + 1
        self.dropout_probs = np.insert(self.dropout_probs, layer_pos, drop_prob)
        return True

    def remove_layer(self, layer_pos):
        """
        Removes the layer in the position received by parameter from the network.
        
        :param layer_pos: Position of the layer that is going to be removed.
        :return: True if the action is done or False if it can not be done.
        """
        if self.number_hidden_layers <= 1:
            return False 
        
        self.dims = np.delete(self.dims, layer_pos)
        self.init_functions = np.delete(self.init_functions, layer_pos)
        self.act_functions = np.delete(self.act_functions, layer_pos)
        self.dropout_probs = np.delete(self.dropout_probs, layer_pos)

        # Finally the number of hidden layers is updated
        self.number_hidden_layers = self.number_hidden_layers - 1
        return True

    def change_layer_dimension(self, layer_pos, new_dim):
        """
        Changes the dimension of the layer in position given by parameter.
        
        :param layer_pos: Position of the layer that will be changed.
        :param new_dim: Dimension that will be changed to.
        :return: True if the action is done or False if it can not be done.
        """
        if new_dim < MIN_NUM_NEURONS or new_dim > self.max_num_neurons:
            return False # It is not within feasible bounds
        self.dims[layer_pos] = new_dim
        return True # This mutation is controled, it always will be applied


    def codify_components(self):
        """
        Codifies all the components of the network in a list of tuples, where the 
        first element of the tuple is a description of the value that is in the 
        second element of the tuple.
        
        :return: List with all the components of the network.
        """
        
        return [('Number of layers', self.number_hidden_layers),
                ('Input dimension', self.input_dim),
                ('Output dimension', self.output_dim),
                ('Neurons in each layer', self.dims),
                ('Initialization functions', self.init_functions),
                ('Activation functions', self.act_functions),
                ('Dropout', self.dropout),
                ('Dropout probabilities', self.dropout_probs),
                ('Batch normalization', self.batch_norm)
               ]

    def __name__(self):
        return 'Multi Layer Perceptron descriptor'


class CNNDescriptor(NetworkDescriptor):
    
    """
    Descriptor of a Convolutional Neural Network, formed by convolutional and pooling layers and 
    with the possibility of selecting the filters and strides in each layer and including 
    batch normalization, but not including dropout.
    
    :param number_hidden_layers: Number of hidden layers in the network.
    :param input_dim: Dimension of the input data (can have one or more dimensions).
    :param output_dim: Expected output of the network (similarly, can have one or more dimensions).
    :param init_functions: Weight initialization functions. Can have different ranges, depending on the subclass.
    :param act_functions: Activation functions to be applied after each layer.
    :param batch_norm: A boolean that indicates if batch normalization is applied after each layer in the network.
    :param dropout: A boolean that indicates if dropout is applied after each layer in the network.
    :param dropout_probs: The different probabilities of the dropout after each layer.

    :param layers: List of indeces that reference the type of layer.
    :param filters: List of filters sizes in each layer.
    :param strides: List of strides sizes in each layer.
    :param max_filter: Maximum size that filters can have.
    :param max_stride: Maximum size that stride can have.
    """
    
    def __init__(self, number_hidden_layers=None, input_dim=None, output_dim=None, 
                 layer_types=[], max_filter=None, max_stride=None, filters=[], strides=[], 
                 init_functions=[], act_functions=[], batch_norm=False):

        super().__init__(number_hidden_layers=number_hidden_layers, input_dim=input_dim, output_dim=output_dim, 
                         init_functions=init_functions, act_functions=act_functions, 
                         dropout=False, dropout_probs=[], batch_norm=batch_norm)
        self.layers = layer_types
        self.filters = filters
        self.strides = strides
        self.max_filter = max_filter
        self.max_stride = max_stride

    def random_init(self, input_size, output_size, max_num_layers, max_num_neurons, max_stride, max_filter, dropout, batch_norm):
        """
        Given the input, the output and some limits for the parametes of the 
        network, a random initialization of the object (the network descriptor) is done. 
        
        :param input_size: Input size of the network.
        :param output_size: Output size of the network.
        :param max_num_layers: Maximum number of layers that can be in the network.
        :param max_num_neurons: Maximum number fo units that can be in each recurrent layer of the network.
        :param max_stride: Maximum stride possible (used as 2).
        :param max_filter: Maximum filter size possible (used as 3).
        :param dropout: A boolean value that indicates if the networks can have dropout.
        :param batch_norm: A boolean value that indicates if the networks can have batch normalization.
        """
        
        self.max_num_layers = max_num_layers
        self.max_num_neurons = max_num_neurons
        
        self.input_dim = input_size
        self.output_dim = output_size

        self.number_hidden_layers = np.random.randint(max_num_layers)+1
        self.layers = []
        self.max_stride = max_stride
        self.strides = []
        self.max_filter = max_filter
        self.filters = []
        self.init_functions = []
        self.act_functions = []
        
        i = 0
        while i < (self.number_hidden_layers - 1):
            new_layer = np.random.choice([0, 1, 2])
            
            self.layers += [2]
            self.filters += [np.array([np.random.randint(MIN_NUM_FILTERS, max_filter)] * 2 + 
                                       [np.random.randint(MIN_NUM_CHANNELS, MAX_NUM_CHANNELS)])]
            self.strides += [np.array([np.random.randint(MIN_NUM_STRIDES, max_stride)] * 2 + [1])]
            shape = calculate_CNN_shape(self.input_dim, self.filters, self.strides, -1)
            if shape[0] < 2 or shape[1] < 2 or (shape[0] < self.output_dim[0] and shape[1] < self.output_dim[1]):  
                # If the blob size is too small
                self.layers = self.layers[:-1]
                self.filters = self.filters[:-1]
                self.strides = self.strides[:-1]
                if i == 0:
                    continue # If enters here is because it has not added anything, 
                             # it tries again with other random
                else:
                    break
            
            self.init_functions += [np.random.choice(initializations)]
            self.act_functions += [np.random.choice(activations)]
            
            if new_layer != 2:
                self.layers += [new_layer]
                self.filters += [np.array([np.random.randint(MIN_NUM_FILTERS, max_filter)] * 2 + 
                                           [np.random.randint(MIN_NUM_CHANNELS, MAX_NUM_CHANNELS)])]
                self.strides += [np.array([np.random.randint(MIN_NUM_STRIDES, max_stride)] * 2 + [1])]
                shape = calculate_CNN_shape(self.input_dim, self.filters, self.strides, -1)
                if shape[0] < 2 or shape[1] < 2 or (shape[0] < self.output_dim[0] and shape[1] < self.output_dim[1]):  
                    # If the blob size is too small
                    self.layers = self.layers[:-2]
                    self.filters = self.filters[:-2]
                    self.strides = self.strides[:-2]
                    self.init_functions = self.init_functions[:-1]
                    self.act_functions = self.act_functions[:-1]
                    if i == 0:
                        continue # If enters here is because it has not added anything, 
                                 # it tries again with other random
                    else:
                        break
                i += 1
                self.init_functions += [np.random.choice(initializations)]
                self.act_functions += [np.random.choice(activations)]
                
            i += 1
        
        self.number_hidden_layers = i
        
        # For the final layer, in order to achieve the desired output, calculations
        # have to been made.
        last_lay_inp = calculate_CNN_shape(self.input_dim, self.filters, self.strides, -1)
        maximum_stride = last_lay_inp[0]//self.output_dim[0]
        if maximum_stride > 1:
            last_lay_stride = np.random.randint(1, maximum_stride)
        else:
            last_lay_stride  = 1
        last_lay_filter = (last_lay_inp[0] + 1) - (self.output_dim[0] * last_lay_stride)
           
        if last_lay_filter < 1:
            return False
        
        self.layers += [2]
        self.filters += [np.array([last_lay_filter, last_lay_filter, self.output_dim[2]])]
        self.strides += [np.array([last_lay_stride, last_lay_stride, 1])]
        self.init_functions += [np.random.choice(initializations)]
        self.act_functions += [np.random.choice(activations)]
        self.number_hidden_layers += 1
        
        self.dropout = False
        self.dropout_probs = np.zeros(self.number_hidden_layers)
        
        if batch_norm is not None and batch_norm:
            self.batch_norm = np.random.choice([True, False])
        else:
            self.batch_norm = False

    
    def add_layer(self, layer_pos, lay_type, filter_size, filter_channel, stride_size, act_function, init_function):
        """
        Adds a layer in the given position with all the characteristics indicated by parameters.
        
        :param layer_pos: Position of the layer.
        :param lay_type: Type of operation (0, 1: pooling, 2 convolutional).
        :param filter_size: Filter size.
        :param stride_size: Stride size.
        :param act_function: Activation function.
        :param init_function: Initialization function.
        :return: True if the action is done or False if it can not be done.
        """
        if self.number_hidden_layers >= self.max_num_layers:
            return False  
        if filter_size < MIN_NUM_FILTERS or filter_size > self.max_filter:
            return False # It is not within feasible bounds
        if filter_channel < MIN_NUM_CHANNELS or filter_channel > MAX_NUM_CHANNELS:
            return False # It is not within feasible bounds
        if stride_size < MIN_NUM_STRIDES or stride_size > self.max_stride:
            return False # It is not within feasible bounds
        
        aux_filters = copy.deepcopy(self.filters)
        aux_strides = copy.deepcopy(self.strides)
        aux_number_hidden_layers = self.number_hidden_layers
        
        if self.layers[layer_pos] != 2:
            layer_pos += 1
        
        aux_number_hidden_layers += 1
        aux_filters.insert(layer_pos, np.array([filter_size, filter_size, filter_channel]))
        aux_strides.insert(layer_pos, np.array([stride_size, stride_size, 1]))
        
        if lay_type < 2:
            aux_number_hidden_layers += 1
            pool_filters = np.random.randint(MIN_NUM_FILTERS, self.max_filter)
            pool_channels = np.random.randint(MIN_NUM_CHANNELS, MAX_NUM_CHANNELS)
            pool_strides = 1
            
            aux_filters.insert(layer_pos+1, np.array([pool_filters, pool_filters, pool_channels]))
            aux_strides.insert(layer_pos+1, np.array([pool_strides, pool_strides, 1]))
            
        
        # The final layer, in order to achieve the desired output, calculations
        # have to been made.
        last_lay_inp = calculate_CNN_shape(self.input_dim, aux_filters[:-1], aux_strides[:-1], -1)
        maximum_stride = last_lay_inp[0]//self.output_dim[0]
        if maximum_stride > 1:
            last_lay_stride = np.random.randint(1, maximum_stride)
        else:
            last_lay_stride  = 1
        last_lay_filter = (last_lay_inp[0] + 1) - (self.output_dim[0] * last_lay_stride)
           
        if last_lay_filter < 1:
            return False
        
        aux_filters[-1] = np.array([last_lay_filter, last_lay_filter, self.output_dim[2]])
        aux_strides[-1] = np.array([last_lay_stride, last_lay_stride, 1])
            
        if aux_number_hidden_layers >= self.max_num_layers:
            return False
            
        if calculate_CNN_shape(self.input_dim, aux_filters, aux_strides, -1)[0] >= self.output_dim[0]:
        
            self.layers.insert(layer_pos, 2)
            # A convolutional layer is added always
            self.act_functions.insert(layer_pos, act_function)
            self.init_functions.insert(layer_pos, init_function)
    
            if lay_type < 2:
                # If a max_pool or avg_pool has to be added
                self.layers.insert(layer_pos+1, lay_type)
                
                self.act_functions.insert(layer_pos+1, None)
                self.init_functions.insert(layer_pos+1, None)               
                
            self.filters = aux_filters
            self.strides = aux_strides
            self.number_hidden_layers = aux_number_hidden_layers           
            
            return True
        else:
            return False

    def remove_layer(self, layer_pos):
        """
        Removes the layer in the position received by parameter from the network.
        
        :param layer_pos: Position of the layer that is going to be removed.
        :return: True if the action is done or False if it can not be done.
        """
        if self.number_hidden_layers <= 1:
            return False # If there is only one layer it can  not be removed
        if self.number_hidden_layers == 2 and self.layers[-1] != 2 and layer_pos == 0:
            # If there are only two layers (an the last one is a pooling layers)
            # If the layer to be removed is the first one, it can not be removed
            # because it can not only remain a pooling layer
            return False

        self.layers.pop(layer_pos)
        self.act_functions.pop(layer_pos)
        self.init_functions.pop(layer_pos)
        self.filters.pop(layer_pos)
        self.strides.pop(layer_pos)
        self.number_hidden_layers -= 1
        
        if layer_pos < self.number_hidden_layers: 
            if self.layers[layer_pos] != 2:
                # If next layers is a pooling one, it has to be removed
                # with its convolutional layer, that has been removed
                self.layers.pop(layer_pos)
                self.act_functions.pop(layer_pos)
                self.init_functions.pop(layer_pos)
                self.filters.pop(layer_pos)
                self.strides.pop(layer_pos)
                self.number_hidden_layers -= 1
        
        # The final layer, in order to achieve the desired output, calculations
        # have to been made.
        last_lay_inp = calculate_CNN_shape(self.input_dim, self.filters[:-1], self.strides[:-1], -1)
        maximum_stride = last_lay_inp[0]//self.output_dim[0]
        if maximum_stride > 1:
            last_lay_stride = np.random.randint(1, maximum_stride)
        else:
            last_lay_stride  = 1
        last_lay_filter = (last_lay_inp[0] + 1) - (self.output_dim[0] * last_lay_stride)
           
        if last_lay_filter < 1:
            return False
        
        self.filters[-1] = np.array([last_lay_filter, last_lay_filter, self.output_dim[2]])
        self.strides[-1] = np.array([last_lay_stride, last_lay_stride, 1])
        
        return True

    def change_filters(self, layer_pos, new_filter_size, new_channel):
        """
        Changes the filter of the layer in the position received by parameter for 
        a new filter specified with the size and channels received also by parameters.
        
        :param layer_pos: Position of the filter to be changed.
        :param new_filter_size: Height and width of the filter (only square filters are allowed).
        :param new_channel: Number of output channels.
        :return: True if the action is done or False if it can not be done.
        """
        if new_filter_size < MIN_NUM_FILTERS or new_filter_size > self.max_filter:
            return False # It is not within feasible bounds
        if new_channel < MIN_NUM_CHANNELS or new_channel > MAX_NUM_CHANNELS:
            return False # It is not within feasible bounds
        
        aux_filters = copy.deepcopy(self.filters)
        aux_strides = copy.deepcopy(self.strides)
        aux_filters[layer_pos][0] = new_filter_size
        aux_filters[layer_pos][1] = new_filter_size
        aux_filters[layer_pos][2] = new_channel
        
        
        # The final layer, in order to achieve the desired output, calculations
        # have to been made.
        last_lay_inp = calculate_CNN_shape(self.input_dim, aux_filters[:-1], aux_strides[:-1], -1)
        maximum_stride = last_lay_inp[0]//self.output_dim[0]
        if maximum_stride > 1:
            last_lay_stride = np.random.randint(1, maximum_stride)
        else:
            last_lay_stride  = 1
        last_lay_filter = (last_lay_inp[0] + 1) - (self.output_dim[0] * last_lay_stride)
           
        if last_lay_filter < 1:
            return False
        
        aux_filters[-1] = np.array([last_lay_filter, last_lay_filter, self.output_dim[2]])
        aux_strides[-1] = np.array([last_lay_stride, last_lay_stride, 1])
        
        if calculate_CNN_shape(self.input_dim, aux_filters, aux_strides, -1)[0] >= self.output_dim[0]:
            self.filters = aux_filters
            self.strides = aux_strides
            return True
        else:
            return False

    def change_stride(self, layer_pos, new_stride_size):
        """
        Changes the stride of the layer in the position received by parameter 
        for a new stried specified with the data received also by parameters.
        
        :param layer_pos: Position of the filter to be changed.
        :param new_stride_size: Stride assigned to that layer. 
        :return: True if the action is done or False if it can not be done.
        """
        if new_stride_size < MIN_NUM_STRIDES or new_stride_size > self.max_stride:
            return False # It is not within feasible bounds
        
        aux_filters = copy.deepcopy(self.filters)
        aux_strides = copy.deepcopy(self.strides)
        aux_strides[layer_pos][0] = new_stride_size
        aux_strides[layer_pos][1] = new_stride_size
        
        # The final layer, in order to achieve the desired output, calculations
        # have to been made.
        last_lay_inp = calculate_CNN_shape(self.input_dim, aux_filters[:-1], aux_strides[:-1], -1)
        maximum_stride = last_lay_inp[0]//self.output_dim[0]
        if maximum_stride > 1:
            last_lay_stride = np.random.randint(1, maximum_stride)
        else:
            last_lay_stride  = 1
        last_lay_filter = (last_lay_inp[0] + 1) - (self.output_dim[0] * last_lay_stride)
           
        if last_lay_filter < 1:
            return False
        
        aux_filters[-1] = np.array([last_lay_filter, last_lay_filter, self.output_dim[2]])
        aux_strides[-1] = np.array([last_lay_stride, last_lay_stride, 1])
        
        if calculate_CNN_shape(self.input_dim, self.filters, self.strides, -1)[0] >= self.output_dim[0]:
            self.filters = aux_filters
            self.strides = aux_strides
            return True
        else:
            return False
            
        
    def codify_components(self):
        """
        Codifies all the components of the network in a list of tuples, where the 
        first element of the tuple is a description of the value that is in the 
        second element of the tuple.
        
        :return: List with all the components of the network.
        """
        
        return [('Number of layers', self.number_hidden_layers),
                ('Input dimension', self.input_dim[0]),
                ('Output dimension', self.output_dim[0]),
                ('Layer types', self.layers),
                ('Filters in each layer', self.filters),
                ('Strides in each layer', self.strides),
                ('Maximum filter size', self.max_filter),
                ('Maximum stride size', self.max_stride),
                ('Initialization functions', self.init_functions),
                ('Activation functions', self.act_functions),
                ('Dropout', self.dropout),
                ('Dropout probabilities', self.dropout_probs),
                ('Batch normalization', self.batch_norm)
               ]
        
    def __name__(self):
        return 'Convolutional Neural Network descriptor'

class TCNNDescriptor(NetworkDescriptor):

    """
    Descriptor of a Transposed Convolutional Neural Network, formed by transposed convolutional 
    layers and with the possibility of selecting the filters and strides in each layer and including 
    batch normalization, but not including dropout.
    
    :param number_hidden_layers: Number of hidden layers in the network.
    :param input_dim: Dimension of the input data (can have one or more dimensions).
    :param output_dim: Expected output of the network (similarly, can have one or more dimensions).
    :param init_functions: Weight initialization functions. Can have different ranges, depending on the subclass.
    :param act_functions: Activation functions to be applied after each layer.
    :param batch_norm: A boolean that indicates if batch normalization is applied after each layer in the network.
    :param dropout: A boolean that indicates if dropout is applied after each layer in the network.
    :param dropout_probs: The different probabilities of the dropout after each layer.

    :param filters: List of filters sizes in each layer.
    :param strides: List of strides sizes in each layer.
    :param max_filter: Maximum size that filters can have.
    :param max_stride: Maximum size that stride can have.
    """
    
    def __init__(self, number_hidden_layers=None, input_dim=None, output_dim=None, 
                 max_filter=None, max_stride=None, filters=[], strides=[], 
                 init_functions=[], act_functions=[], batch_norm=False):
        super().__init__(number_hidden_layers=number_hidden_layers, input_dim=input_dim, output_dim=output_dim, 
                         init_functions=init_functions, act_functions=act_functions, 
                         dropout=False, dropout_probs=[], batch_norm=batch_norm)
        self.filters = filters
        self.strides = strides
        self.max_filter = max_filter
        self.max_stride = max_stride

    def random_init(self, input_size, output_size, max_num_layers, max_num_neurons, max_stride, max_filter, dropout, batch_norm):
        """
        Given the input, the output and some limits for the parametes of the 
        network, a random initialization of the object (the network descriptor) is done. 
        
        :param input_size: Input size of the network.
        :param output_size: Output size of the network.
        :param max_num_layers: Maximum number of layers that can be in the network.
        :param max_num_neurons: Maximum number fo units that can be in each recurrent layer of the network.
        :param max_stride: Maximum stride possible (used as 2).
        :param max_filter: Maximum filter size possible (used as 3).
        :param dropout: A boolean value that indicates if the networks can have dropout.
        :param batch_norm: A boolean value that indicates if the networks can have batch normalization.
        """
        
        self.max_num_layers = max_num_layers
        self.max_num_neurons = max_num_neurons
        
        self.input_dim = input_size
        self.output_dim = output_size
        
        self.max_stride = max_stride
        self.strides = []
        self.max_filter = max_filter
        self.filters = []
        self.init_functions = []
        self.act_functions = []
        for i in range(300):
            self.filters += [np.array([np.random.randint(MIN_NUM_FILTERS, max_filter)] * 2 + 
                                       [np.random.randint(MIN_NUM_CHANNELS, MAX_NUM_CHANNELS)])]
            self.strides += [np.array([np.random.randint(MIN_NUM_STRIDES, max_stride)] * 2 + [1])]
            self.init_functions += [np.random.choice(initializations)]
            self.act_functions += [np.random.choice(activations)]

            shape = calculate_TCNN_shape(self.input_dim, self.filters, self.strides, -1)
            
            # Once the expected shape is exceeded, we have enough layers
            if shape[0] >= self.output_dim[0] and shape[1] >= self.output_dim[1]:  
                prev_shape = calculate_TCNN_shape(self.input_dim, self.filters[:-1], self.strides[:-1], -1)
                while prev_shape[0] * self.strides[-1][0] > self.output_dim[0]:
                    self.strides[-1] = np.array([self.strides[-1][0] - 1, self.strides[-1][1] - 1, self.strides[-1][2]])
                desired_filter_size = self.output_dim[0] - (prev_shape[0] - 1) * self.strides[-1][0]
                self.filters[-1] = np.array([desired_filter_size, desired_filter_size, self.output_dim[2]])
                self.number_hidden_layers = i+1
                break

        if batch_norm is not None and not batch_norm:
            self.batch_norm = np.random.choice([True, False])
                
    def add_layer(self, layer_pos, filter_size, filter_channel, stride_size, act_function, init_function):
        """
        Adds a layer in the given position with all the characteristics indicated by parameters.
        
        :param layer_pos: Position of the layer.
        :param filter_size: Filter size.
        :param stride_size: Stride size.
        :param act_function: Activation function.
        :param init_function: Initialization function.
        :return: True if the action is done or False if it can not be done.
        """
        if self.number_hidden_layers >= self.max_num_layers:
            return False 
        if filter_size < MIN_NUM_FILTERS or filter_size > self.max_filter:
            return False # It is not within feasible bounds
        if filter_channel < MIN_NUM_CHANNELS or filter_channel > MAX_NUM_CHANNELS:
            return False # It is not within feasible bounds
        if stride_size < MIN_NUM_STRIDES or stride_size > self.max_stride:
            return False # It is not within feasible bounds
        
        self.number_hidden_layers += 1
        self.filters.insert(layer_pos, np.array([filter_size, filter_size, filter_channel]))
        self.strides.insert(layer_pos, np.array([stride_size, stride_size, 1]))
        self.act_functions.insert(layer_pos, act_function)
        self.init_functions.insert(layer_pos, init_function)
            
        for i in range(layer_pos, self.number_hidden_layers):
            
            if i > self.number_hidden_layers - 1: #I the layers have been removed
                break
            
            output = calculate_TCNN_shape(self.input_dim, self.filters[:i], self.strides[:i], -1)
            
            while output[0] * self.strides[i][0] >= self.output_dim[0]:
                self.strides[i] = np.array([self.strides[i][0] - 1, self.strides[i][1] - 1, self.strides[i][2]])
            if self.strides[i][0] == 0:
                self.remove_layer(i)
            
        output = calculate_TCNN_shape(self.input_dim, self.filters[:-1], self.strides[:-1], -1)
        desired_filter_size = self.output_dim[0] - (output[0] - 1) * self.strides[-1][0]
        self.filters[-1] = np.array([desired_filter_size, desired_filter_size, self.output_dim[2]])
        
        return True

    def remove_layer(self, layer_pos):
        """
        Removes the layer in the position received by parameter from the network.
        
        :param layer_pos: Position of the layer that is going to be removed.
        :return: True if the action is done or False if it can not be done.
        """
        if self.number_hidden_layers <= 1:
            return False 
        
        self.filters.pop(layer_pos)
        self.act_functions.pop(layer_pos)
        self.init_functions.pop(layer_pos)
        self.strides.pop(layer_pos)
        self.number_hidden_layers -= 1
        
        output = calculate_TCNN_shape(self.input_dim, self.filters[:-1], self.strides[:-1], -1)
        
        while output[0] * self.strides[-1][0] > self.output_dim[0]:
            self.strides[-1] = np.array([self.strides[-1][0] - 1, self.strides[-1][1] - 1, self.strides[-1][2]])
        
        if self.strides[-1][0] == 0:
            if not self.remove_layer(-1):
                return False
        
        output = calculate_TCNN_shape(self.input_dim, self.filters[:-1], self.strides[:-1], -1)
        desired_filter_size = self.output_dim[0] - (output[0] - 1) * self.strides[-1][0]
        self.filters[-1] = np.array([desired_filter_size, desired_filter_size, self.output_dim[2]])
        return True

    def change_filters(self, layer_pos, new_filter_size, new_channel):
        """
        Changes the filter of the layer in the position received by parameter for 
        a new filter specified with the size and channels received also by parameters.
        
        :param layer_pos: Position of the filter to be changed.
        :param new_filter_size: Height and width of the filter (only square filters are allowed).
        :param new_channel: Number of output channels.
        :return: True if the action is done or False if it can not be done.
        """
        if new_filter_size < MIN_NUM_FILTERS or new_filter_size > self.max_filter:
            return False # It is not within feasible bounds
        if new_channel < MIN_NUM_CHANNELS or new_channel > MAX_NUM_CHANNELS:
            return False # It is not within feasible bounds
        
        self.filters[layer_pos][0] = new_filter_size
        self.filters[layer_pos][1] = new_filter_size
        self.filters[layer_pos][2] = new_channel
        
        i = layer_pos
        while i < self.number_hidden_layers:
            output = calculate_TCNN_shape(self.input_dim, self.filters[:i], self.strides[:i], -1)
            
            while output[0] * self.strides[i][0] > self.output_dim[0]:
                self.strides[i] = np.array([self.strides[i][0] - 1, self.strides[i][1] - 1, self.strides[i][2]])
 
            if self.strides[i][0] == 0:
                if not self.remove_layer(i):
                    return False
            i += 1

        output = calculate_TCNN_shape(self.input_dim, self.filters[:-1], self.strides[:-1], -1)
        desired_filter_size = self.output_dim[0] - (output[0] - 1) * self.strides[-1][0]
        self.filters[-1] = np.array([desired_filter_size, desired_filter_size, self.output_dim[2]])
        
        return True

    def change_stride(self, layer_pos, new_stride_size):
        """
        Changes the stride of the layer in the position received by parameter 
        for a new stried specified with the data received also by parameters.
        
        :param layer_pos: Position of the stride to be changed.
        :param new_stride_size: Stride assigned to that layer.
        :return: True if the action is done or False if it can not be done.
        """
        if new_stride_size < MIN_NUM_STRIDES or new_stride_size > self.max_stride:
            return False # It is not within feasible bounds
        
        self.strides[layer_pos][0] = new_stride_size
        self.strides[layer_pos][1] = new_stride_size

        i = layer_pos
        while i < self.number_hidden_layers:
            output = calculate_TCNN_shape(self.input_dim, self.filters[:i], self.strides[:i], -1)

            while output[0] * self.strides[i][0] > self.output_dim[0]:
                self.strides[i] = np.array([self.strides[i][0] - 1, self.strides[i][1] - 1, self.strides[i][2]])

            if self.strides[i][0] == 0:
                if not self.remove_layer(i):
                    return False
            i += 1

        output = calculate_TCNN_shape(self.input_dim, self.filters[:-1], self.strides[:-1], -1)
        desired_filter_size = self.output_dim[0] - (output[0] - 1) * self.strides[-1][0]
        self.filters[-1] = np.array([desired_filter_size, desired_filter_size, self.output_dim[2]])

        return True
        
    def codify_components(self):
        """
        Codifies all the components of the network in a list of tuples, where the 
        first element of the tuple is a description of the value that is in the 
        second element of the tuple.
        
        :return: List with all the components of the network.
        """
        
        return [('Number of layers', self.number_hidden_layers),
                ('Input dimension', self.input_dim[0]),
                ('Output dimension', self.output_dim[0]),
                ('Filters in each layer', self.filters),
                ('Strides in each layer', self.strides),
                ('Maximum filter size', self.max_filter),
                ('Maximum stride size', self.max_stride),
                ('Initialization functions', self.init_functions),
                ('Activation functions', self.act_functions),
                ('Dropout', self.dropout),
                ('Dropout probabilities', self.dropout_probs),
                ('Batch normalization', self.batch_norm)
               ]
        
    def __name__(self):
        return 'Transposed Convolutional Neural Network descriptor'

class RNNDescriptor(NetworkDescriptor):
    """
    Descriptor of a Recurrent Neural Network, formed by recurrent layers and with
    the possibility of selecting the type and number of unit in the recurrent
    layer and allowing bidirectional layers.
    
    :param number_hidden_layers: Number of hidden layers in the network.
    :param input_dim: Dimension of the input data (can have one or more dimensions).
    :param output_dim: Expected output of the network (similarly, can have one or more dimensions).
    :param init_functions: Weight initialization functions. Can have different ranges, depending on the subclass.
    :param act_functions: Activation functions to be applied after each layer.
    :param batch_norm: A boolean that indicates if batch normalization is applied after each layer in the network.
    :param dropout: A boolean that indicates if dropout is applied after each layer in the network.
    :param dropout_probs: The different probabilities of the dropout after each layer.

    :param rnn_layers: List of the types of recurrent layers.
    :param units_in_layer: List of number of units in each layer.
    :param max_units: Maximum numbers of units that layers can have.
    :param bidirectional: A boolean value to indicate if bidirectional layers are allowed.
    """
    
    def __init__(self, number_hidden_layers=None, input_dim=None, output_dim=None, 
                 rnn_layers = [], bidirectional=[], units_in_layer=[], max_units=None,
                 init_functions=[], act_functions=[], 
                 dropout=False, dropout_prob=[], batch_norm=False):
        
        super().__init__(number_hidden_layers=number_hidden_layers, input_dim=input_dim, output_dim=output_dim, 
             init_functions=init_functions, act_functions=act_functions, dropout=dropout, batch_norm=batch_norm)
        self.rnn_layers = rnn_layers
        self.units_in_layer = [min(unit, max_units) for unit in units_in_layer]
        self.max_units = max_units
        self.bidirectional = bidirectional
        
    def random_init(self, input_size, output_size, max_num_layers, max_num_neurons, max_stride, max_filter, dropout, batch_norm):
        """
        Given the input, the output and some limits for the parametes of the 
        network, a random initialization of the object (the network descriptor) is done. 
        
        :param input_size: Input size of the network.
        :param output_size: Output size of the network.
        :param max_num_layers: Maximum number of layers that can be in the network.
        :param max_num_neurons: Maximum number fo units that can be in each recurrent layer of the network.
        :param max_stride: Maximum stride possible (used as 2).
        :param max_filter: Maximum filter size possible (used as 3).
        :param dropout: A boolean value that indicates if the networks can have dropout.
        :param batch_norm: A boolean value that indicates if the networks can have batch normalization.
        """
        
        self.max_num_layers = max_num_layers
        self.max_num_neurons = max_num_neurons
        
        self.input_dim = input_size
        self.output_dim = output_size
        
        self.max_units = max_num_neurons
        
        # Random initialization
        self.number_hidden_layers = np.random.randint(max_num_layers)+1
        self.units_in_layer = [np.random.randint(MIN_NUM_NEURONS, self.max_units) for _ in range(self.number_hidden_layers)]
        self.init_functions = list(np.random.choice(initializations, size=self.number_hidden_layers))
        self.act_functions = list(np.random.choice(activations, size=self.number_hidden_layers))
        
        self.rnn_layers = list(np.random.choice([SimpleRNN, LSTM, GRU], size=self.number_hidden_layers))
        self.bidirectional = list(np.random.choice([True, False], size=self.number_hidden_layers))
        
        if dropout is not None and dropout:
            self.dropout = np.random.choice([True, False])
            self.dropout_probs = np.random.rand(self.number_hidden_layers)
        else:
            self.dropout_probs = np.zeros(self.number_hidden_layers)
    
    def add_layer(self, layer_pos, rnn_type, units_in_layer, bidirectional, act_function, init_function):
        """
        Adds a layer in the given position with all the characteristics indicated by parameters.
        
        :param layer_pos: Position of the layer.
        :param lay_params: Type of recurrent layer, how many units, etc.
        :return: True if the action is done or False if it can not be done.
        """
        
        if self.number_hidden_layers >= self.max_num_layers:
            return False 
        if units_in_layer < MIN_NUM_NEURONS or units_in_layer > self.max_num_neurons:
            return False # It is not within feasible bounds
        
        self.number_hidden_layers += 1
        self.rnn_layers.insert(layer_pos, rnn_type)
        self.units_in_layer.insert(layer_pos, min(units_in_layer, self.max_units))
        self.bidirectional.insert(layer_pos, bidirectional)
        self.act_functions.insert(layer_pos, act_function)
        self.init_functions.insert(layer_pos, init_function)
        
        return True
    
    def remove_layer(self, layer_pos):
        """
        Removes the layer in the position received by parameter from the network.
        
        :param layer_pos: Position of the layer that is going to be removed.
        :return: True if the action is done or False if it can not be done.
        """

        if self.number_hidden_layers <= 1:
            return False 
        
        self.number_hidden_layers -= 1
        self.rnn_layers.pop(layer_pos)
        self.units_in_layer.pop(layer_pos)
        self.bidirectional.pop(layer_pos)
        self.act_functions.pop(layer_pos)
        self.init_functions.pop(layer_pos)
        
        return True
    
    def change_layer_type(self, layer_pos, layer_type): 
        """
        Changes the type of the layer in the position received by parameter 
        for a new one selected randomly.
        
        :param layer_pos: Position of the filter to be changed.
        :return: True if the action is done or False if it can not be done.
        """
        self.rnn_layers[layer_pos] = layer_type
        return True
        
    def change_units(self, layer_pos, new_units):
        """
        Changes the number of units of the layer in the position received by parameter 
        for a new number specified with the data received also by parameters.
        
        :param layer_pos: Position of the filter to be changed.
        :param new_units: Number of units assigned to that layer. 
        :return: True if the action is done or False if it can not be done.
        """
        if new_units < MIN_NUM_NEURONS or new_units > self.max_num_neurons:
            return False # It is not within feasible bounds
        
        self.units_in_layer[layer_pos] = new_units
        return True
        
    def change_bidirectional(self, layer_pos):
        """
        Changes the layer in the position received by parameter 
        for a bidirectional if is not already; otherwise, for a non
        bidirectional one.
        
        :param layer_pos: Position of the filter to be changed.
        :return: True if the action is done or False if it can not be done.
        """
        self.bidirectional[layer_pos] = not self.bidirectional[layer_pos]
        return True
    
    def codify_components(self):
        """
        Codifies all the components of the network in a list of tuples, where the 
        first element of the tuple is a description of the value that is in the 
        second element of the tuple.
        
        :return: List with all the components of the network.
        """
        
        return [('Number of layers', self.number_hidden_layers),
                ('Input dimension', self.input_dim[0]),
                ('Output dimension', self.output_dim[0]),
                ('Layer types', self.rnn_layers),
                ('Maximum number of units', self.max_units),
                ('Units in each layer', self.units_in_layer),
                ('Bidirectional layers are allowed', self.bidirectional),
                ('Initialization functions', self.init_functions),
                ('Activation functions', self.act_functions),
                ('Dropout', self.dropout),
                ('Dropout probabilities', self.dropout_probs),
                ('Batch normalization', self.batch_norm)
               ]
        
    def __name__(self):
        return 'Recurrent Neural Network descriptor'
    
class Network:
    """
    This class contains the TensorFlow definition of the networks (i.e., the "implementation" of the descriptors).
    
    :param network_descriptor: The descriptor with the information to generate this class (the network).
    """
    def __init__(self, network_descriptor):
        self.descriptor = network_descriptor

class MLP(Network):
    """
    TensorFlow definition of the Multi Layer Perceptron.
    
    :param network_descriptor: Descriptor of the MLP.
    """
    def __init__(self, network_descriptor):
        super().__init__(network_descriptor)

    def building(self, x):
        """
        Given a TensorFlow layer, this functions continues adding more layers of a MLP.
        
        :param x: A layer from TensorFlow.
        :return: The layer received from parameter with the MLP concatenated to it.
        """
        
        for lay_indx in range(self.descriptor.number_hidden_layers-1):
            
            x = Dense(self.descriptor.dims[lay_indx], 
                      activation=self.descriptor.act_functions[lay_indx], 
                      kernel_initializer=self.descriptor.init_functions[lay_indx])(x)
            if self.descriptor.dropout:
                x = Dropout(self.descriptor.dropout_probs[lay_indx])(x)
            if self.descriptor.batch_norm:
                x = BatchNormalization()(x)
        
        x = Dense(self.descriptor.output_dim, 
                  activation=self.descriptor.act_functions[self.descriptor.number_hidden_layers-1],
                  kernel_initializer=self.descriptor.init_functions[self.descriptor.number_hidden_layers-1])(x)
        if self.descriptor.dropout:
            x = Dropout(self.descriptor.dropout_probs[self.descriptor.number_hidden_layers-1])(x)
        if self.descriptor.batch_norm:
            x = BatchNormalization()(x)
        
        return x


class CNN(Network):
    """
    TensorFlow definition of the Convolutional Neural Network.
    
    :param network_descriptor: Descriptor of the CNN.
    """
    def __init__(self, network_descriptor):
        super().__init__(network_descriptor)

    def building(self, x):
        """
        Given a TensorFlow layer, this functions continues adding more layers of a CNN.
        
        :param x: A layer from TensorFlow.
        :return: The layer received from parameter with the CNN concatenated to it.
        """
        
        lay_indx = 0
        while lay_indx < self.descriptor.number_hidden_layers:
            
            x = Conv2D(self.descriptor.filters[lay_indx][2],
                       [self.descriptor.filters[lay_indx][0],self.descriptor.filters[lay_indx][1]],
                       strides=[self.descriptor.strides[lay_indx][0], self.descriptor.strides[lay_indx][1]],
                       padding="valid",
                       activation=self.descriptor.act_functions[lay_indx],
                       kernel_initializer=self.descriptor.init_functions[lay_indx])(x)
            
            if self.descriptor.batch_norm:
                x = BatchNormalization()(x)

            lay_indx += 1

            if lay_indx < self.descriptor.number_hidden_layers:
               
                if self.descriptor.layers[lay_indx] == 0:  # If is has average pooling
                    x = AveragePooling2D(pool_size=[self.descriptor.filters[lay_indx][0], self.descriptor.filters[lay_indx][1]],
                                               strides=[self.descriptor.strides[lay_indx][0], self.descriptor.strides[lay_indx][1]],
                                               padding="valid")(x)
                    lay_indx += 1
                elif self.descriptor.layers[lay_indx] == 1: # If it has max pooling
                    x = MaxPooling2D(pool_size=[self.descriptor.filters[lay_indx][0], self.descriptor.filters[lay_indx][1]],
                                           strides=[self.descriptor.strides[lay_indx][0], self.descriptor.strides[lay_indx][1]],
                                           padding="valid")(x)
                    lay_indx += 1
                
        return x


class TCNN(Network):
    """
    TensorFlow definition of the Transposed Convolutional Neural Network.
    
    :param network_descriptor: Descriptor of the TCNN.
    """
    def __init__(self, network_descriptor):
        super().__init__(network_descriptor)

    def building(self, x):
        """
        Given a TensorFlow layer, this functions continues adding more layers of a TCNN.
        
        :param x: A layer from TensorFlow.
        :return: The layer received from parameter with the TCNN concatenated to it.
        """       
        for lay_indx in range(self.descriptor.number_hidden_layers):
            
            x = Conv2DTranspose(self.descriptor.filters[lay_indx][2],
                                      [self.descriptor.filters[lay_indx][0],self.descriptor.filters[lay_indx][1]],
                                      strides=[self.descriptor.strides[lay_indx][0], self.descriptor.strides[lay_indx][1]],
                                      padding="valid",
                                      activation=self.descriptor.act_functions[lay_indx],
                                      kernel_initializer=self.descriptor.init_functions[lay_indx])(x)

        return x

class RNN(Network):
    """
    TensorFlow definition of the Recurrent Neural Network.
    
    :param network_descriptor: Descriptor of the RNN.
    """
    def __init__(self, network_descriptor):
        super().__init__(network_descriptor)

    def building(self, x):
        """
        Given a TensorFlow layer, this functions continues adding more layers of a RNN.
        
        :param x: A layer from TensorFlow.
        :return: The layer received from parameter with the RNN concatenated to it.
        """
        for lay_indx in range(self.descriptor.number_hidden_layers - 1):
            
            rnn_layer = self.descriptor.rnn_layers[lay_indx](units=self.descriptor.units_in_layer[lay_indx],
                                                             return_sequences=True,
                                                             activation=self.descriptor.act_functions[lay_indx],
                                                             kernel_initializer=self.descriptor.init_functions[lay_indx]())
            
            if self.descriptor.bidirectional[lay_indx]:
                x = Bidirectional(rnn_layer)(x)
            else:
                x = rnn_layer(x)
        
            if self.descriptor.dropout:
                x = Dropout(self.descriptor.dropout_probs[lay_indx])(x)
                
        rnn_layer = self.descriptor.rnn_layers[self.descriptor.number_hidden_layers - 1](
                        units=self.descriptor.units_in_layer[self.descriptor.number_hidden_layers - 1],
                        return_sequences=False,
                        activation=self.descriptor.act_functions[self.descriptor.number_hidden_layers - 1],
                        kernel_initializer=self.descriptor.init_functions[self.descriptor.number_hidden_layers - 1]())
        
        if self.descriptor.bidirectional[self.descriptor.number_hidden_layers - 1]:
            x = Bidirectional(rnn_layer)(x)
        else:
            x = rnn_layer(x)
            
        return x

def calculate_CNN_shape(input_shape, filters, strides, desired_layer):
    """
    Given the input shape and the filters and strides that have been applied to that
    input by using convolutional and pooling layers, it return the output shape 
    in the desired layer.
    
    :param input_shape: Shape of the input given to the convolutional layers.
    :param filters: List with the filters that convolutional and pooling layers use.
    :param strides: List with the strides that convolutional and pooling layers use.
    :param desired_layer: Position of the layer where the output shape is wanted (if 
                          -1 is passed, it calculates until the last layer). 
        
    :return: The output shape after applying those operations.
    """
    if desired_layer == -1:
        return calculate_CNN_shape(input_shape, filters, strides, len(filters))
    if desired_layer == 0:
        return input_shape
    
    filter_size = filters[0]
    stride_size = strides[0]
    output_shape = (np.array(input_shape[:2]) - np.array(filter_size[:2])) // np.array(stride_size[:2]) + 1
    return calculate_CNN_shape(output_shape, filters[1:], strides[1:], desired_layer-1)


def calculate_TCNN_shape(input_shape, filters, strides, desired_layer):
    """
    Given the input shape and the filters and strides that have been applied to that
    input by using transposed convolutional layers, it return the output shape 
    in the desired layer.
    
    :param input_shape: Shape of the input given to the transposed convolutional layers.
    :param filters: List with the filters that transposed convolutional layers use.
    :param strides: List with the strides that transposed convolutional layers use.
    :param desired_layer: Position of the layer where the output shape is wanted (if 
                          -1 is passed, it calculates until the last layer). 
        
    :return: The output shape after applying those operations.
    """
    if desired_layer == -1:
        return calculate_TCNN_shape(input_shape, filters, strides, len(filters))
    if desired_layer == 0:
        return input_shape
    
    filter_size = filters[0]
    stride_size = strides[0]
    output_shape = [input_shape[0] * stride_size[0] + max(filter_size[0] - stride_size[0], 0), 
                    input_shape[1] * stride_size[1] + max(filter_size[1] - stride_size[1], 0)]
    return calculate_TCNN_shape(output_shape, filters[1:], strides[1:], desired_layer-1)
