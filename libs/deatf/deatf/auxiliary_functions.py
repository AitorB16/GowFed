"""
Here are the auxiliary functions that are used in the rest of the programs. These functions 
can be divided in three groups:
    
    1. A function for dividing the data in batches.
    2. Some functions for evaluation predicted values returned by the network.
    3. Functions to load data (fashion and mnist).
    
========================================================================================================
"""

from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np

def batch(data, n, i):
    """
    Extract from the given data a batch that go from the position i
    to the i + n; being n the batch size .
    
    :param data: Set of solutions intended to be fed to the network.
    :param n: Size of the desired batch. 
    :param i: Index of the last solution used in the last epoch.
    :return: The batch of data form x with size n since the index i of the data.
    """

    if i + n > data.shape[0]:  
        # In case there are not enough solutions before the end of the array
        index = i + n-data.shape[0]  # Select all the individuals until the end and restart
        return np.concatenate((data[i:, :], data[:index, :]))
    else:  
        # Easy case
        index = i+n
        return data[i:index, :]


def mse(true, prediction):
    """
    Calculates the mean squared error with numpy functions.
    
    :param true: True data, is the ground of truth in the evaluation.
    :param prediction: Predicted data, is the data that is evaluated.
    :return: The mean squared error calculated between the true and predicted data.
    """
    return np.sum((true-prediction)**2)/true.shape[0]

def accuracy_error(true, prediction):
    """
    Calculates the accuracy error with numpy functions.
    
    :param true: True data, is the ground of truth in the evaluation.
    :param prediction: Predicted data, is the data that is evaluated.
    :return: The accuracy error calculated between the true and predicted data.
    """
    if len(true.shape) > 1:
        true = np.argmax(true, axis=-1)
    if len(prediction.shape) > 1:
        prediction = np.argmax(prediction, axis=-1)
    
    return 1 - np.sum(true == prediction)/np.prod(true.shape)

def balanced_accuracy(true, prediction):
    """
    Calculates the balanced accuracy with numpy functions.
    
    :param true: True data, is the ground of truth in the evaluation.
    :param prediction: Predicted data, is the data that is evaluated.
    :return: The balaced accuracy error calculated between the true and predicted data.
    """
    # Number of classes (n), and the number of examples per class are computed.
    classes, count = np.unique(true, return_counts=True)
    # Weights for each class are computed. Summation of the weights of all examples belonging to a class must be 1/n.
    class_weights = [1/len(classes)/i for i in count]
    # Weights for each example are computed, depending on their class.
    example_weights = [class_weights[i] for i in true]
    # Accuracy is computed weighting each example according to the representation of the class it belongs to in the data.
    return accuracy_score(true, prediction, sample_weight=example_weights)


def load_fashion():
    """
    Loads and returns the data from the fashion mnist dataset and is returned already
    divided in train, validation and test.
    
    :return: Data of fashion mnist dataset divided in train, test and validation
             (x_train, y_train, x_test, y_test, x_val, y_val).
    :rtype x_train: uint8 NumPy array of grayscale image data with shapes (42000, 28, 28), 
                    containing the training data.    
    :rtype y_train: uint8 NumPy array of labels (integers in range 0-9) with shape (42000,) 
                    for the training data.    
    :rtype x_test: uint8 NumPy array of grayscale image data with shapes (10000, 28, 28), 
                   containing the test data.
    :rtype y_test: uint8 NumPy array of labels (integers in range 0-9) with shape (10000,) 
                   for the test data.
    :rtype x_val: uint8 NumPy array of grayscale image data with shapes (18000, 28, 28), 
                  containing the validation data.
    :rtype y_val: uint8 NumPy array of labels (integers in range 0-9) with shape (18000,) 
                  for the validation data.    
    """
    fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()

    dividing_indx = int(fashion_mnist[0][0].shape[0] * 0.7)
    
    x_train = fashion_mnist[0][0][:dividing_indx]
    y_train = fashion_mnist[0][1][:dividing_indx]

    x_val = fashion_mnist[0][0][dividing_indx:]
    y_val = fashion_mnist[0][1][dividing_indx:]

    x_test = fashion_mnist[1][0]
    y_test = fashion_mnist[1][1]

    return x_train, y_train, x_test, y_test, x_val, y_val

def load_mnist():
    """
    Loads and returns the data from the mnist dataset and is returned already
    divided in train, validation and test.
    
    :return: Data of mnist dataset divided in train, test and validation
             (x_train, y_train, x_test, y_test, x_val, y_val).
    :rtype x_train: uint8 NumPy array of grayscale image data with shapes (42000, 28, 28), 
                    containing the training data. Pixel values range from 0 to 255.   
    :rtype y_train: uint8 NumPy array of labels (integers in range 0-9) with shape (42000,) 
                    for the training data.    
    :rtype x_test: uint8 NumPy array of grayscale image data with shapes (10000, 28, 28), 
                   containing the test data. Pixel values range from 0 to 255.
    :rtype y_test: uint8 NumPy array of labels (integers in range 0-9) with shape (10000,)
                   for the test data.
    :rtype x_val: uint8 NumPy array of grayscale image data with shapes (18000, 28, 28), 
                  containing the validation data. Pixel values range from 0 to 255.
    :rtype y_val: uint8 NumPy array of labels (integers in range 0-9) with shape (18000,) 
                  for the validation data.    
    """
    
    mnist = tf.keras.datasets.mnist.load_data()
 
    dividing_indx = int(mnist[0][0].shape[0] * 0.7)
    
    x_train = mnist[0][0][:dividing_indx]
    y_train = mnist[0][1][:dividing_indx]

    x_val = mnist[0][0][dividing_indx:]
    y_val = mnist[0][1][dividing_indx:]
    
    x_test = mnist[1][0]
    y_test = mnist[1][1]

    return x_train, y_train, x_test, y_test, x_val, y_val