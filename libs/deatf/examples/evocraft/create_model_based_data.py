import numpy as np

import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal, RandomUniform, GlorotNormal, GlorotUniform

activations = [None, tf.nn.relu, tf.nn.elu, tf.nn.softplus, tf.nn.softsign, tf.sigmoid, tf.nn.tanh]
initializations = [RandomNormal, RandomUniform, GlorotNormal, GlorotUniform]

blocks_activation = dict()
for i,a in enumerate(activations):
	blocks_activation[a] = i
	
blocks_initialization = dict()
for i,a in enumerate(initializations):
	blocks_initialization[a] = i

positions_neurons = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,3),(2,0),(2,3),(3,0),(3,1),(3,2),(3,3)]

def creating_data_and_population(evolving_alg, wanted_blocks, bounds, center_layer, layer_size):
    """
    Creates data and noisy data based in the network descriptors that are received as 
    parameter. That data will be the representation of the network in a build form.
    
    :param evolving alg: Evolving algorithm that will be used in the GA process.
    :param wanted_blocks: List with the desired blocks for the builds.
    :param bounds: Bounds where the build can be created.
    :param center_layer: Coordinates of the center where the build wil be created.
    :param layer_size: Size of the layers from the network.
    :return: Created data based in the network descriptors, noisy data and population
             of toolbox.
    """
    population = evolving_alg.toolbox.population(n=evolving_alg.population_size)
    
    network_descriptors = [ind.descriptor_list for ind in population]
    data = generate_data(network_descriptors, wanted_blocks, bounds, center_layer, layer_size)
    noisy_data = add_noise(data, wanted_blocks)
    
    return data, noisy_data, population

def generate_data(network_descriptors, wanted_blocks, bounds, center_layer, layer_size):
    """
    Creates data based in the network descriptors that are received as parameter. That
    data will be the representation of the network in a build form.
    
    :param network_descriptors: List with the network descriptors that will be used
                                to create the data.
    :param wanted_blocks: List with the desired blocks for the builds.
    :param bounds: Bounds where the build can be created.
    :param center_layer: Coordinates of the center where the build wil be created.
    :param layer_size: Size of the layers from the network.
    :return: Created data based in the network descriptors.
    """
    data = []
    
    for desc in network_descriptors:
        mlp_descriptor = desc['n0']
        
        # Each network is represented in a cube of the same dimensions of the bound
        # each layer of the cube represents a hidden layer of the network
        # they are generated containing the air block
        cube = np.full((bounds),-1).astype(int)  
        
        # For each layer in the descriptor, we first get the index of the corresponding component (activation-function, initialization,etc)
        # And then we associate this index to a minecraft block
        # Each cube_layer gets in the center the four codes for the main activation functions and in the sides
        # it associates a block for each neuron. Since there might be less neurons than free positions (12)
        # the positions associated to each neuron are randomly selected. 
        for i in range(mlp_descriptor.number_hidden_layers):
            cube_layer = np.full((layer_size,layer_size),-1).astype(int)  # Each layer of the cube is initialized full of air
            cube_layer[center_layer,center_layer] = wanted_blocks[blocks_activation[mlp_descriptor.act_functions[i]]]
            cube_layer[center_layer,center_layer+1] = wanted_blocks[blocks_initialization[mlp_descriptor.init_functions[i]]+1]
            cube_layer[center_layer+1,center_layer] = wanted_blocks[int(mlp_descriptor.dropout)]
            cube_layer[center_layer+1,center_layer+1] = wanted_blocks[int(mlp_descriptor.batch_norm)]
            n_neurons = mlp_descriptor.dims[i]
        	
            #random_perm_indices = np.random.permutation(max_neurons)
            random_indices = np.random.choice(range(len(positions_neurons)), size=n_neurons, replace=False)
            for j in random_indices:
                cube_layer[positions_neurons[j]] = 64 ##wanted_blocks[np.random.randint(n_wanted_blocks-1)+1]
            #The cube layer is save in the block    
            cube[:,i,:]=cube_layer
        	
            #print(i,cube_layer)
        data.append(cube.flatten())
    return np.array(data)

def add_noise(data, wanted_blocks):
    """
    Add noise to given data. Values that are different from -1
    are changed to it. In minecraft and cube representation, this 
    means to remove a block from a position.
    
    :param data: Data created from the network descriptorss.
    :param wanted_blocks: List with teh desired blocks in the build.
    :return: Received data but with noise added to it.
    """
    data_blocks_indices = np.where(data != -1)
    data_blocks_indices = np.array(data_blocks_indices).transpose()
    
    num_blocks = data_blocks_indices.shape[1]
    
    data_air_indices = np.where(data == -1)
    data_air_indices = np.array(data_air_indices).transpose()
    
    num_blocks = data_blocks_indices.shape[0]
    num_airs = data_air_indices.shape[0]
    
    num_noisy_blocks = np.random.randint(int(num_blocks*0.1),int(num_blocks*0.4))
    num_noisy_air = np.random.randint(int(num_airs*0.1),int(num_airs*0.4))
    
    noisy_data = np.copy(data)
    
    for _ in range(num_noisy_blocks):
        random_indx = np.random.randint(data_blocks_indices.shape[0])
        random_block_indices = data_blocks_indices[random_indx]
        noisy_data[random_block_indices[0],random_block_indices[1]] = -1
        data_blocks_indices = np.delete(data_blocks_indices, random_indx, 0)
        
    for _ in range(num_noisy_air):
        random_indx = np.random.randint(data_air_indices.shape[0])
        random_air_indices = data_air_indices[random_indx]
        noisy_data[random_air_indices[0],random_air_indices[1]] = np.random.choice(wanted_blocks)
        data_air_indices = np.delete(data_air_indices, random_indx, 0)
    
    return noisy_data