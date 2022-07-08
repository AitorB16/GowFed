"""
Mutation is a key step in the genetic algorithm. Here are defined and explained all
the classes used to make those mutations possible. :class:`~Mutation` is the main and
generic class; then there are specific classes, one for each :class:`deatf.network.NetworkDescriptor`.
These classes contains all the possible mutations that can be made in their defined functions.
But the two steps to follow are:
    
1. Initialize appropiate class for the Network to evolve.
2. Call :func:`Mutation.apply_random_mutation` in order to apply a random possible mutation
   or :func:`Mutation.apply_mutation` to apply an specific mutation (these functions 
   are inherited from Mutation so they can be used in other mutation classes).

The mutation will be applied to the :class:`deatf.network.NetworkDescriptor` object passed
i nthe parameter network in the initialization. The change will be made in the object, that
is why mutation function do not return anything.

========================================================================================================
"""

import numpy as np
from deatf.network import initializations, activations
from deatf.network import MIN_NUM_NEURONS, MIN_NUM_FILTERS, MIN_NUM_STRIDES, MIN_NUM_CHANNELS, MAX_NUM_CHANNELS
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU

class Mutation:
    """
    This class implements the possible mutations that can be applied for a generic
    networks descriptors. This class contains the general mutations that could be 
    applied to any descriptor, subclases will have specific mutations for their own
    network.
    
    :param hypers: List with the hyperparameters to be evolved.
    :param batch_norm: A boolean value that indicates if batch normalization can appear
                       during evolution. Is different from the network's one because this
                       remains during all the process and the network's one can be changed 
                       or mutated.
    :param dropout: A boolean value that indicates the same as batch normalization but 
                    applied to dropout.
    :param network: Network descriptor that will be mutated.
    :param hyperparameters: Hyperparameters that are in the mutated network.
    :param custom_mutation_list: Optinal list with the mutations that can be applied
                                 to the network. Is defined by the user with the mutations
                                 wanted, if is not defined, all possible mutations could be applied.
    """
    def __init__(self, hypers, batch_norm, dropout, network, hyperparameters, custom_mutation_list=[]):
        self.hypers = hypers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.network = network
        self.hyperparameters = hyperparameters
        if not custom_mutation_list:
            custom_mutation_list = self.methods()
        self.custom_mutation_list = custom_mutation_list

    def apply_random_mutation(self):
        """
        It selects all the available mutations for the network and after selecting
        one randomly, it applies that mutation.
        """
        possible_mutations = self.custom_mutation_list[:]
        type_mutation = np.random.choice(possible_mutations)  
        while not self.apply_mutation(type_mutation):
            possible_mutations.pop(possible_mutations.index(type_mutation))
            type_mutation = np.random.choice(possible_mutations)
        
    def apply_mutation(self, mutation):
        """
        Given a mutation method and a network, it applies that mutation to the 
        network if it is possible; if not it raises a ValueError. This function
        is mainly though for the case where the user define the mutation list, in
        order to ensure that all the described mutation can be done.
        
        :param mutation: Mutation that will be applied to the network.
        """
        if mutation in self.methods():
            return eval('self.' + mutation + '()')
        else:
            raise ValueError('The mutation {} is not defined'.format(mutation))
        
    def methods(self):
        """
        Finds and returns all the possible mutations for that mutation class.
        
        :return: List with all the possible mutations that can be applied.
        """
        method_list = [method for method in dir(self.__class__) if not method.startswith('__')]
        
        method_list.remove('methods')
        method_list.remove('apply_mutation')
        method_list.remove('apply_random_mutation')
        
        
        if len(self.hypers) == 0:
            method_list.remove('mut_hyper')
        if not self.batch_norm: # Batch normalization declared in evolution is used
            method_list.remove('mut_batch_norm')
        if not self.dropout: # Dropout declared in evolution is used
            method_list.remove('mut_dropout')
        
        return method_list
       
    def mut_weight_init(self):
        """
        Changes the weight initialization in a random layer for a new random 
        weight initialization function. 
        """
        layer_pos = np.random.randint(self.network.number_hidden_layers)
        
        actual_init = self.network.init_functions[layer_pos]
        possible_initializations = initializations[:]
        possible_initializations.remove(actual_init)

        init_w_function = np.random.choice(possible_initializations)
            
        return self.network.change_weight_init(layer_pos, init_w_function)

    def mut_activation(self):
        """
        Changes the activation function in a random layer for a new random 
        activation function. 
        """
        layer_pos = np.random.randint(self.network.number_hidden_layers)
        
        actual_activ = self.network.act_functions[layer_pos]
        possible_activations = activations[:]
        possible_activations.remove(actual_activ)
        
        init_a_function = np.random.choice(possible_activations)
        
        return self.network.change_activation(layer_pos, init_a_function)
        
    def mut_dropout(self):
        """
        Changes the dropout boolean.
        """
        return self.network.change_dropout()
        
    def mut_dropout_prob(self):
        """
        Changes the dropout probabilites.
        """
        return self.network.change_dropout_prob()

    def mut_batch_norm(self):
        """
        Changes the batch normalization boolean.
        """
        return self.network.change_batch_norm() 
        
    def mut_hyper(self):
        """
        Changes one of the hyperparameters of the network for a new one selected
        randomly from the list of evolvable hyperparameters described in the initialization.
        """
        # We change the value of a hyperparameter for another value
        h = np.random.choice(list(self.hypers.keys()))  # We select the hyperparameter to be mutated
        # We choose two values, just in case the first one is the one already selected
        new_value = np.random.choice(self.hypers[h], size=2, replace=False)
        if self.hyperparameters[h] == new_value[0]:
            self.hyperparameters[h] = new_value[1]
        else:
            self.hyperparameters[h] = new_value[0]
            
        return True # This mutation is controled, it always will be applied
           
            
class MLP_Mutation(Mutation):
    """
    This class implements the possible mutations that can be applied 
    specialy to the MLP descriptors. 
    
    :param hypers: List with the hyperparameters to be evolved.
    :param batch_norm: A boolean value that indicates if batch normalization can appear
                       during evolution. Is different from the network's one because this
                       remains during all the process and the network's one can be changed 
                       or mutated.
    :param dropout: A boolean value that indicates the same as batch normalization but 
                    applied to dropout.possible_initializations
    :param network: MLP descriptor that will be mutated.
    :param hyperparameters: Hyperparameters that are in the mutated network.
    :param custom_mutation_list: Optinal list with the mutations that can be applied
                                 to the network. Is defined by the user with the mutations
                                 wanted, if is not defined, all possible mutations could be applied.
    """
    def __init__(self, hypers, batch_norm, dropout, network, hyperparameters, custom_mutation_list=[]):
        super().__init__(hypers, batch_norm, dropout, network, hyperparameters, custom_mutation_list)
        
    def mut_add_layer(self):
        """
        A new dense layer is added to the MLP descriptor.
        """
        new_layer_pos = np.random.randint(self.network.number_hidden_layers)
        new_lay_dims = np.random.randint(MIN_NUM_NEURONS, self.network.max_num_neurons)
        new_act_function = np.random.choice(activations)
        new_init_function = np.random.choice(initializations)
        new_drop_prob = np.random.rand()
    
        return self.network.add_layer(new_layer_pos, new_lay_dims, new_init_function, 
                                      new_act_function, new_drop_prob)
    
    def mut_remove_layer(self): 
        """
        Removes a random layer from the MLP descriptor.
        """
        return self.network.remove_random_layer()

    def mut_dimension(self):
        """
        Number of neurons of a random layer are changed to a new random
        number of neurons from 1 to maximum number of neurons described 
        in the initialization.
        """
        layer_pos = np.random.randint(self.network.number_hidden_layers)
        
        possible_dimensions = list(range(MIN_NUM_NEURONS, self.network.max_num_neurons))
        possible_dimensions.remove(self.network.dims[layer_pos])
        
        if possible_dimensions:
            new_dim = np.random.choice(possible_dimensions)
        else:
            return False # No mutation can be done
        
        return self.network.change_layer_dimension(layer_pos, new_dim)
     
        
class CNN_Mutation(Mutation):
    """
    This class implements the possible mutations that can be applied 
    specialy to the CNN descriptors. 
    
    :param hypers: List with the hyperparameters to be evolved.
    :param batch_norm: A boolean value that indicates if batch normalization can appear
                       during evolution. Is different from the network's one because this
                       remains during all the process and the network's one can be changed 
                       or mutated.
    :param dropout: A boolean value that indicates the same as batch normalization but 
                    applied to dropout.
    :param network: CNN descriptor that will be mutated.
    :param hyperparameters: Hyperparameters that are in the mutated network.
    :param custom_mutation_list: Optinal list with the mutations that can be applied
                                 to the network. Is defined by the user with the mutations
                                 wanted, if is not defined, all possible mutations could be applied.
    """
    def __init__(self, hypers, batch_norm, dropout, network, hyperparameters, custom_mutation_list=[]):
        super().__init__(hypers, batch_norm, dropout, network, hyperparameters, custom_mutation_list)

    def mut_add_conv_layer(self):
        """
        A new convolutional or pooling layer is added to the CNN descriptor.
        """
        if self.network.number_hidden_layers <= 1:
            new_layer_pos = 0
        else:
            # Is self.network.number_hidden_layers - 1 because last layer should
            # not be changed in order to have the desired output.
            new_layer_pos = np.random.randint(self.network.number_hidden_layers-1)
        new_lay_type = np.random.randint(3)
        
        new_filter_size = np.random.randint(MIN_NUM_FILTERS, self.network.max_filter)
        new_filter_channel = np.random.randint(MIN_NUM_CHANNELS, MAX_NUM_CHANNELS)
        new_stride_size = np.random.randint(MIN_NUM_STRIDES, self.network.max_stride)
        new_act_function = np.random.choice(activations)
        new_init_function = np.random.choice(initializations)
        
        return self.network.add_layer(new_layer_pos, new_lay_type, new_filter_size, new_filter_channel,
                               new_stride_size, new_act_function, new_init_function)
        
    def mut_remove_conv_layer(self):
        """
        Removes a random layer from the CNN descriptor if it is possible (if there is only
        one layer, it can not be removed).        
        """
        return self.network.remove_random_layer()
    
    def mut_stride_conv(self):
        """
        Stride of a random layer is changed to a new random
        stride from 1 to maximum stride size described 
        in the network descriptor.
        """
        if self.network.number_hidden_layers <= 1:
            return False
        # Is self.network.number_hidden_layers - 1 because last layer should
        # not be changed in order to have the desired output.
        layer_pos = np.random.randint(self.network.number_hidden_layers-1)
        
        possible_strides = list(range(MIN_NUM_STRIDES, self.network.max_stride))
        possible_strides.remove(self.network.strides[layer_pos][0])
        
        if possible_strides:
            new_stride = np.random.choice(possible_strides)
        else:
            return False # No mutation can be done
        return self.network.change_stride(layer_pos, new_stride)
        
    def mut_filter_conv(self):
        """
        Filter of a random layer is changed to a new random
        filter with size from 2 to maximum filter size described 
        in the network descriptor and with channels from 1 to 65.
        """
        if self.network.number_hidden_layers <= 1:
            return False
        # Is self.network.number_hidden_layers - 1 because last layer should
        # not be changed in order to have the desired output.
        layer_pos = np.random.randint(self.network.number_hidden_layers-1)
        
        possible_filter_sizes = list(range(MIN_NUM_FILTERS, self.network.max_filter))
        possible_filter_channels = list(range(MIN_NUM_CHANNELS, MAX_NUM_CHANNELS))
        possible_filter_sizes.remove(self.network.filters[layer_pos][0])
        possible_filter_channels.remove(self.network.filters[layer_pos][2])
        
        if possible_filter_channels and possible_filter_sizes:
            new_filter_size = np.random.choice(possible_filter_sizes)
            channels = np.random.choice(possible_filter_channels)
        else:
            return False # No mutation can be done
        
        return self.network.change_filters(layer_pos, new_filter_size, channels)
     
    def mut_weight_init(self):
        """
        Changes the weight initialization in a random layer for a new random 
        weight initialization function. 
        """
        layer_pos = np.random.randint(self.network.number_hidden_layers)
        if self.network.layers[layer_pos] != 2:
            # If it is a pooling layer, has it does not have a weight 
            # initialization, layer_index will be recalculated to point
            # the convolutional layer of the pooling layer.
            layer_pos -= 1
        
        actual_init = self.network.init_functions[layer_pos]
        possible_initializations = initializations[:]
        possible_initializations.remove(actual_init)

        init_w_function = np.random.choice(possible_initializations)
            
        return self.network.change_weight_init(layer_pos, init_w_function)

    def mut_activation(self):
        """
        Changes the activation function in a random layer for a new random 
        activation function. 
        """
        layer_pos = np.random.randint(self.network.number_hidden_layers)
        if self.network.layers[layer_pos] != 2:
            # If it is a pooling layer, has it does not have a activation 
            # function, layer_index will be recalculated to point
            # the convolutional layer of the pooling layer.
            layer_pos -= 1
            
        actual_activ = self.network.act_functions[layer_pos]
        possible_activations = activations[:]
        possible_activations.remove(actual_activ)
        
        init_a_function = np.random.choice(possible_activations)
        
        return self.network.change_activation(layer_pos, init_a_function)

class TCNN_Mutation(Mutation):
    """
    This class implements the possible mutations that can be applied 
    specialy to the TCNN descriptors. 
    
    :param hypers: List with the hyperparameters to be evolved.
    :param batch_norm: A boolean value that indicates if batch normalization can appear
                       during evolution. Is different from the network's one because this
                       remains during all the process and the network's one can be changed 
                       or mutated.
    :param dropout: A boolean value that indicates the same as batch normalization but 
                    applied to dropout.
    :param network: TCNN descriptor that will be mutated.
    :param hyperparameters: Hyperparameters that are in the mutated network.
    :param custom_mutation_list: Optinal list with the mutations that can be applied
                                 to the network. Is defined by the user with the mutations
                                 wanted, if is not defined, all possible mutations could be applied.
    """
    def __init__(self, hypers, batch_norm, dropout, network, hyperparameters, custom_mutation_list=[]):
        super().__init__(hypers, batch_norm, dropout, network, hyperparameters, custom_mutation_list)
         
    def mut_add_deconv_layer(self):
        """
        A new transposed convolutional layer is added to the TCNN descriptor.
        """
        new_layer_pos = np.random.randint(self.network.number_hidden_layers)
        
        new_filter_size = np.random.randint(MIN_NUM_FILTERS, self.network.max_filter)
        new_filter_channel = np.random.randint(MIN_NUM_CHANNELS, MAX_NUM_CHANNELS)
        new_stride_size = np.random.randint(MIN_NUM_STRIDES, self.network.max_stride)
        new_act_function = np.random.choice(activations)
        new_init_function = np.random.choice(initializations)
        
        return self.network.add_layer(new_layer_pos, new_filter_size, new_filter_channel,
                                      new_stride_size, new_act_function, new_init_function)
    
    def mut_remove_deconv_layer(self):
        """
        Removes a random layer from the TCNN descriptor if it is possible (if there is only
        one layer, it can not be removed).        
        """
        return self.network.remove_random_layer()
    
    def mut_stride_deconv(self):
        """
        Stride of a random layer is changed to a new random
        stride from 1 to maximum stride size described 
        in the network descriptor.
        """
        if self.network.number_hidden_layers <= 1:
            return False
        # Is self.network.number_hidden_layers - 1 because last layer should
        # not be changed in order to have the desired output.
        layer_pos = np.random.randint(self.network.number_hidden_layers - 1)
        
        possible_strides = list(range(MIN_NUM_STRIDES, self.network.max_stride))
        possible_strides.remove(self.network.strides[layer_pos][0])
        
        if possible_strides:
            new_stride = np.random.choice(possible_strides)
        else:
            return False # No mutation can be done
        
        return self.network.change_stride(layer_pos, new_stride)
    
    def mut_filter_deconv(self):
        """
        Filter of a random layer is changed to a new random
        filter with size from 2 to maximum filter size described 
        in the network descriptor and with channels from 1 to 65.
        """
        if self.network.number_hidden_layers <= 1:
            return False
        # Is self.network.number_hidden_layers - 1 because last layer should
        # not be changed in order to have the desired output.
        layer_pos = np.random.randint(self.network.number_hidden_layers - 1)
        
        possible_filter_sizes = list(range(MIN_NUM_FILTERS, self.network.max_filter))
        possible_filter_channels = list(range(MIN_NUM_CHANNELS, MAX_NUM_CHANNELS))
        possible_filter_sizes.remove(self.network.filters[layer_pos][0])
        possible_filter_channels.remove(self.network.filters[layer_pos][2])
        
        if possible_filter_sizes and possible_filter_channels:
            new_filter_size = np.random.choice(possible_filter_sizes)
            channels = np.random.choice(possible_filter_channels)
        else:
            return False # No mutation can be done
        
        return self.network.change_filters(layer_pos, new_filter_size, channels)

class RNN_Mutation(Mutation):
    """
    This class implements the possible mutations that can be applied 
    specialy to the RNN descriptors. 
    
    :param hypers: List with the hyperparameters to be evolved.
    :param batch_norm: A boolean value that indicates if batch normalization can appear
                       during evolution. Is different from the network's one because this
                       remains during all the process and the network's one can be changed 
                       or mutated.
    :param dropout: A boolean value that indicates the same as batch normalization but 
                    applied to dropout.
    :param network: RNN descriptor that will be mutated.
    :param hyperparameters: Hyperparameters that are in the mutated network.
    :param custom_mutation_list: Optinal list with the mutations that can be applied
                                 to the network. Is defined by the user with the mutations
                                 wanted, if is not defined, all possible mutations could be applied.
    """
    def __init__(self, hypers, batch_norm, dropout, network, hyperparameters, custom_mutation_list=[]):
        super().__init__(hypers, batch_norm, dropout, network, hyperparameters, custom_mutation_list)
        
    def mut_add_rnn_layer(self):
        """
        A new recurrent layer is added to the RNN descriptor.        
        """
        layer_pos = np.random.randint(self.network.number_hidden_layers)
        
        rnn_type = np.random.choice([SimpleRNN, LSTM, GRU])
        units_in_layer = np.random.randint(MIN_NUM_NEURONS, self.network.max_units)
        bidirectional = np.random.choice([True, False])
        act_function = np.random.choice(activations)
        init_function = np.random.choice(initializations)
        
        return self.network.add_layer(layer_pos, rnn_type, units_in_layer, bidirectional, act_function, init_function)
        
    def mut_remove_rnn_layer(self):
        """
        Removes a random layer from the RNN descriptor.   
        """
        return self.network.remove_random_layer()

    def mut_change_layer_type(self):
        """
        Type of a random layer is changed to a new random type.
        """
        layer_pos = np.random.randint(self.network.number_hidden_layers)
        
        possible_types = [SimpleRNN, LSTM, GRU]
        possible_types.remove(self.network.rnn_layers[layer_pos])
        
        layer_type = np.random.choice(possible_types)
        
        return self.network.change_layer_type(layer_pos, layer_type)
        
    def mut_change_units(self):
        """
        Number of units in a random layer is changed to a new random 
        number of units from 1 to maximum number of units described in 
        network descriptor.        
        """
        
        layer_pos = np.random.randint(self.network.number_hidden_layers)
        
        possible_units = list(range(MIN_NUM_NEURONS, self.network.max_units))
        possible_units.remove(self.network.units_in_layer[layer_pos])
        
        if possible_units:
            new_units = np.random.choice(possible_units)
        else:
            return False # No mutation can be done
        
        return self.network.change_units(layer_pos, new_units)
        
    def mut_change_bidirectional(self):
        """
        A random layer is changed to be biderctional if is not; otherwise,
        it stops beeing bidirectional.        
        """
        layer_pos = np.random.randint(self.network.number_hidden_layers)
        return self.network.change_bidirectional(layer_pos)
