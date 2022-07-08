import tensorflow as tf
import numpy as np

from visualize_networks import show_network


def interactive_loss(probabilities, wanted_blocks, bounds, indx, evaluate=False):
    """
    Function that shows the block that is descriving the probability distribution.
    That build is created with the wanted blocks and with thhe maximum shape of 
    the bounds. It can be used to ask the user and be an evalaution method or simply
    a method to visualize the data in Minecraft.
    
    :param probabilities: Data that will be shown, it contains the probability
                         distribution that describes the build.
    :param wanted_blocks: List with the index of the desired blocks to creait the build.
    :param bounds: Maximum size of the build, it also represent if created buid will
                   have two or three dimensions.
    :param indx: List with the index of a place in the minecraft world where the
                 build will be created.
    :param evaluate: Boolean value that express if the evaluation will be done or
                     just visualization.
                     
    :return: The punctuation of the printed model if evaluation is done, else -1.
    """
    probabilities = tf.math.sigmoid(probabilities)
    blocks = tf.math.argmax(probabilities, -1)
    
    printed_blocks = blocks.numpy()
    for i, val in enumerate(wanted_blocks):
        printed_blocks = np.where(printed_blocks == i,val, printed_blocks)
        
    random_indx = np.random.choice(list(range(len(printed_blocks))))
    #print('rand_indx: ', random_indx)
    
    punctuation = show_network(printed_blocks[random_indx], bounds, [(bounds[0]*2)*indx[0], 10, (bounds[0]*2)*indx[1]], evaluate)
    
    return punctuation