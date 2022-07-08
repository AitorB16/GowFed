import tensorflow as tf

def total_symmetry(probabilities):
    """
    Calculates the symetry in all the axis of the received data.
    
    :param probabilities: Data with the probability distribution that express how the 
                          builds will be created. It has to follow the shape: 
                          (num_samples, bounds, available_blocks). Buounds can be of
                          2 or 3 dimensions.
    :return: Symmetry value cacluated with the received data in all the axis.
    """
    num_axis = len(probabilities.shape)
    sym = 0
    for ax in range(num_axis):
        sym += symmetry(probabilities, ax)
        #print(f'Axis: {ax} \n Symmetry: {sym}')
    sym /= num_axis
    #print(f'Total symettry: {sym}')
    return sym

def symmetry(probabilities, axis):
    """
    Calculates the symetry in the given axis of the received data.
    
    :param probabilities: Data with the probability distribution that express how the 
                          builds will be created. It has to follow the shape: 
                          (num_samples, bounds, available_blocks). Buounds can be of
                          2 or 3 dimensions.
    :param axis: The axis of the data where the symetry will be looked.
    :return: Symmetry loss cacluated with the received data.
    """
    index = int(probabilities.shape[axis]/2)
    #print([index, probabilities.shape[axis]-index, -1])
    #print(probabilities)
    left, middle, right = tf.split(probabilities, [index, probabilities.shape[axis]-2*index, -1], axis=axis)
    #print('Left', left.shape, ' : \n', left)
    #print('Right', right.shape, ' : \n', right)
    right = tf.reverse(right,axis=[axis])
    #print('Reversed Right', right.shape, ' : \n', right)
    difference = tf.keras.losses.categorical_crossentropy(left, right)
    #print(difference)
    result = tf.reduce_mean(difference)
    #print(f'Result: {result}')
    return result
"""
import numpy as np

asa = np.array([np.array([-2.7445993 ,-2.2469692,  -2.5746398 , -2.3782146 ]),
              np.array([-2.4640982  ,-2.2854025,  -2.410407  , -2.6066587 ]),
              np.array([-2.4919333  ,-2.445139 ,  -2.3414588 , -2.3486304 ]),
              np.array([-2.5510526  ,-2.5999348,  -2.0209384 , -2.3761086 ]),
              np.array([-2.5768437  ,-2.5316153,  -2.5613687 , -2.490177  ]),
              np.array([-2.5856366  ,-2.7067447,  -2.585881  , -2.4180267 ])])

available_blocks = len([-1,164, 169, 173])

probabilities = np.argmax(asa,axis=-1)
used_probabilities = np.reshape(probabilities, [-1])
y = np.unique(used_probabilities)
num_used_probabilities = len(y)

assert num_used_probabilities <= available_blocks, 'Something has gone wrong in variance penalty.'
if num_used_probabilities < available_blocks*0.3:
    x = 0.  
else:
    x = num_used_probabilities/available_blocks
    """