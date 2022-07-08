'''
In order to modularize and make easier the testing of the networks, functions
have been generalized and grouped in this file. Many of those functions are
for loading the datasets that are also in this library, the rest of them are
for the testing of the networks.
'''
import math
import sys
sys.path.append('../..')

import pandas as pd
import numpy as np

from deatf.evolution import Evolving

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

import tensorflow_datasets as tfds

def test(dataset_name, descriptors=[], eval_func=None, batch_size=150, population=5, 
         generations=10, iters=100, seed=None, max_filter=4, max_stride=3,
         lrate=0.01, cxp=0, mtp=1, evol_alg='mu_plus_lambda', sel='best', sel_kwargs={},
         max_num_layers=10, max_num_neurons=20, hyperparameters={}, 
         is_time_series=False, series_input_width=30, series_label_width=1):
    """
    Responsible of load the desired dataset, prepare it to fit it into
    the models and it to them. Also it call the evolutionary process in
    order to evolve those generated models.
    
    :param dataset_name: String with the name of one of the available datasets.
    :param descriptors: List with the descriptors that will be used in the test.
    :param eval_func: Evaluation function that will be used to se the performance of the models.
    :param batch_size: Batch size that will be taken form the data during training.
    :param population: Number of individuals that will be evaluated in each generation 
                       of the evolution algorithm.
    :param generations: Number of generations that the evolution algorithm will be executed.
    :param iters: Number of iterations that each model will be trained.
    :param seed: Seed of the random processes.
    :param lrate: Learning rate.
    :param cxp: Crossover probability.
    :param mtp: Mutation probability.
    :param evol_alg: Evolutionary algorithm that will be used (strig or a function).
    :param sel: Selection method that will be used (strig or a function).
    :param max_num_layers: Maximum number of layer allowed in the initialization of the networks.
    :param max_num_neurons: Maximum number of neurons in each layer allowed in the 
                            initialization of the networks. 
    :param hyperparameters: Dictionary with the hyperparameters to be evolved.
    :param is_time_series: Boolean that indicates if the processed data is a time series or not.
    :param series_input_width: If is a series of data, the width that the input data will have. 
    :param series_label_width: If is a series of data, the width that the labels will have.
    :return: The last generation, a log book (stats) and the hall of fame (the best 
                 individuals found).
    """
    
    x_train, x_test, x_val, y_train, y_test, y_val, mode = load_dataset(dataset_name, 
            is_time_series=is_time_series, series_input_width=series_input_width, series_label_width=series_label_width)
    
    if not isinstance(y_train[0], float):
        OHEnc = OneHotEncoder()
    
        y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()
        y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
        y_val = OHEnc.fit_transform(np.reshape(y_val, (-1, 1))).toarray()
        
    else:
        y_train = np.reshape(y_train, (-1, 1))
        y_test = np.reshape(y_test, (-1, 1))
        y_val = np.reshape(y_val, (-1, 1))

    input_shape = x_train.shape[1:]
    output_shape = y_val.shape[1:]
    if len(output_shape) == 1:
        output_shape =  [int(math.sqrt(output_shape[0])) + 1, int(math.sqrt(output_shape[0])) + 1, 1]
    
    if eval_func == None:
        eval_func = select_evaluation(mode)
    
    e = Evolving(evaluation=eval_func, 
			 desc_list=descriptors, 
			 x_trains=[x_train], y_trains=[y_train], 
			 x_tests=[x_val], y_tests=[y_val],
			 n_inputs=[input_shape],
			 n_outputs=[output_shape],
			 batch_size=batch_size,
			 population=population,
			 generations=generations,
			 iters=iters, 
             seed=seed,
             lrate=lrate,
             cxp=cxp,
             mtp=mtp,
             evol_alg=evol_alg,
             sel=sel,
             sel_kwargs=sel_kwargs,
			 max_num_layers=max_num_layers, 
			 max_num_neurons=max_num_neurons,
             max_filter=max_filter,
             max_stride=max_stride,
             hyperparameters=hyperparameters)   
     
    a = e.evolve()
    return a

def select_evaluation(mode):
    """
    Given the mode of the dataset (classification or regression), the loss
    function that has to be used is returned.
    
    :param mode: Mode of the dataset 'class' if classification and 'regr' if 
                 regresion.
    :return: Loss function that sould be used in each case.
    """
    
    if mode == "class":
        loss = "XEntropy"
    if mode == "regr":
        loss = "MSE"
        
    return loss

def load_dataset(dataset_name, is_time_series=False, series_input_width=30, series_label_width=1):
    """
    By receiving the dataset's name is capable of loading and dividing it 
    into train, test and validation.
    
    :param dataset_name: String with the name of one of the available datasets.
    :param is_time_series: Boolean that indicates if the processed data is a time series or not.
    :param series_input_width: If is a series of data, the width that the input data will have. 
    :param series_label_width: If is a series of data, the width that the labels will have.
    :return: Inidcated dataset separated in the following way X_train, X_test, X_val, 
             y_train, y_test, y_val, mode
    """
    
    datasets = {'mushrooms': load_mushrooms,
                'air_quality': load_air_quality,
                'estambul_values': load_estambul_values,
                'forest_fires': load_forest_fires,
                'forest_types': load_forest_types,
                'parkinsons': load_parkinsons
                }
    
    datasets_CNN = ['mnist', 'kmnist', 'cmaterdb', 'fashion_mnist',
                    'binary_alpha_digits', 'cifar10', 'rock_paper_scissors']
    
    if dataset_name in datasets.keys():
        features, labels, mode = datasets[dataset_name]()
        if is_time_series:
            features, labels = transform_to_time_series(features, labels, series_input_width, series_label_width)
    elif dataset_name in datasets_CNN:
        features, labels, mode = load_CNN(dataset_name)
    else:
        raise ValueError('The dataset \'{}\' is not available'.format(dataset_name))
        
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
    
    dividing_indx = int(X_train.shape[0] * 0.7)

    X_val = X_train[dividing_indx:]
    y_val = y_train[dividing_indx:]
    
    X_train = X_train[:dividing_indx]
    y_train = y_train[:dividing_indx]
    
    return X_train, X_test, X_val, y_train, y_test, y_val, mode

def load_mushrooms():   
    '''
    Loads the dataset of mushrooms that is inside this library.
    
    :return: Featurees, labels and the type of problem for the dataset.
    '''
    features, labels = load_csv("./datasets/mushroom/mushroom.csv",'\\t', '.',
                                'class',[])
    labels = [int(l) for l in labels]
    return features, labels, 'class'

def load_air_quality():
    '''
    Loads the dataset of air quality that is inside this library.
    
    :return: Featurees, labels and the type of problem for the dataset.
    '''
    features, labels = load_csv("./datasets/air_quality/AirQualityUCI.csv",';', ',',
                                'RH',['Date','Time','Unnamed: 15','Unnamed: 16'])
    
    labels = list(map(float, labels))
    
    return features, labels, 'regr'

def load_estambul_values():
    '''
    Loads the dataset of estambul values that is inside this library.
    
    :return: Featurees, labels and the type of problem for the dataset.
    '''
    features, labels = load_csv("./datasets/estambul_values/data_akbilgic.csv",',', '.',
                                'ISE',['date','Unnamed: 10','Unnamed: 11'])
    return features, labels, 'regr'

def load_forest_fires():
    '''
    Loads the dataset of forest fires that is inside this library.
    
    :return: Featurees, labels and the type of problem for the dataset.
    '''
    features, labels = load_csv("./datasets/forest_fires/forestfires.csv",',', '.',
                                'area',['month', 'day'])
    return features, labels, 'regr'

def load_forest_types():
    '''
    Loads the dataset of forest types that is inside this library.
    
    :return: Featurees, labels and the type of problem for the dataset.
    '''
    train_features, train_labels = load_csv("./datasets/forest_types/training.csv",',', '.',
                                            'class',[])
    test_features, test_labels = load_csv("./datasets/forest_types/testing.csv",',', '.',
                                            'class',[])
    
    labels = np.concatenate((train_labels, test_labels), axis=0)
    features = np.concatenate((train_features, test_features), axis=0)
    
    labels = list(map(ord, map(lambda x: x.strip(), labels)))
    
    return features, labels, 'class'

def load_parkinsons():
    '''
    Loads the dataset of parkinsons that is inside this library.
    
    :return: Featurees, labels and the type of problem for the dataset.
    '''
    features, labels = load_csv("./datasets/parkinsons/parkinsons.csv", ',', ',',
                                'status',['name'])
    labels = [int(l) for l in labels]
    return features, labels, 'class'

def load_CNN(dataset_name):
    """
    Loads the desired dataset that is supposed to be in the database of TensorFlow
    and is for a CNN. 
    
    :param dataset_name: String with the name of one of the available datasets. 
    :return: Featurees, labels and the type of problem for the dataset.
    """
    
    dataset = tfds.load(dataset_name)
    
    features = []
    labels = []
    
    if 'train' in dataset.keys():
        dataset_train = dataset["train"]
        for example in tfds.as_numpy(dataset_train):
            image, label = example['image'], example['label']
            features.append(image)
            labels.append(label)
            
    if 'test' in dataset.keys():
        dataset_test = dataset["test"]    
        for example in tfds.as_numpy(dataset_test):
            image, label = example['image'], example['label']
            features.append(image)
            labels.append(label) 
        
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels, 'class'

def load_csv(data_directory, data_sep, decimal, label_column, removed_columns):
    """
    Loads data in the csv file located in the recceived direction and that follows 
    the received intructions.
    
    :param data_directory: Directory where the desired dataset is located.
    :param data_sep: Delemiter of data separator in the file.
    :param decimal: The way decimal data is expressed in the file.
    :param label_column: Name of the column that contains the labels.
    :param removed_columns: Columns in the file that are not desired.
    :return: Features and labels of the dataset.
    """
    
    data = pd.read_csv(data_directory, sep=data_sep, decimal=decimal, engine='python')    
    data = data.drop(columns=removed_columns)
    data = data.dropna()

    labels = data[label_column].values

    data = data.drop(columns=label_column)
    features = np.array(data)

    scaler = MinMaxScaler(feature_range=(0, 1))
    features = scaler.fit_transform(features)
    if isinstance(labels[0], float) or isinstance(labels[0], int):
        labels = scaler.fit_transform(labels.reshape(-1, 1)).flatten()
    
    return features, labels

def transform_to_time_series(features, labels, input_width, label_width):
    """
    Transforms the received features and labels into time series with the
    desired width in each case.
    
    :param features: Features of the orignal data.
    :param labels: Labels of the original data.
    :param input_width: Desired width for the input data in the time series.
    :param label_width: Desired width for the labels in the time series.
    :return: Features and labels from the original data transformed into time
             series with the desired widths.
    """
    x, y = [], []
    
    for i in range(len(features)-(input_width+label_width)+1):
        x.append(features[i:i+input_width])
    x = np.array(x)
    for i in range(label_width):
        y.append(labels[i+input_width:len(labels)-label_width+i+input_width-1])
    y = np.array(y)
    y = np.transpose(y)

    return x, y
    

if __name__ == "__main__":
    #features, labels, model = load_air_quality()
    #features, labels, model = load_CNN('cifar10')
    X_train, X_test, X_val, y_train, y_test, y_val, mode = load_dataset('air_quality', True, 30, 1)
    #x,y = transform_to_time_series(features, labels, 30, 1)