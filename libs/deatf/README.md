# DEATF

![Python](https://img.shields.io/badge/Python-3.6-blue)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.5-green)](https://www.tensorflow.org/)
[![DEAP](https://img.shields.io/badge/DEAP-1.0-brightgreen)](https://deap.readthedocs.io/en/master/)

Distributed Evolutionary Algorithms in TensorFlow (DEATF) is a framework where networks generated with <a href="https://www.tensorflow.org/">TensorFlow</a> [[1]](#1) are evolved via <a href="deap.readthedocs.org/">DEAP</a> [[2]](#2). DEATF is a framework directly based in <a href="https://github.com/unaigarciarena/EvoFlow">EvoFlow</a> [[3]](#3) framework created by Unai Garciarena.

![DEATF-logo](./docs/_images/DEATF-logo.png)

## Installation

DEATF has available an easy installation with pip.

```bash
pip install deatf
```

## Documentation

In order to facilitate the use of DEATF, there is a <a href='https://deatf.readthedocs.io/en/latest/index.html'>User's Guide</a> with the information of all classes, functions and examples provided by the library.

## Requirements

DEATF requires a Python version between 3.5 and 3.8. It also depends from other libraries, these are the requirements to use it correctly:

 - TensorFlow (v2.0 or greater)
 -  Numpy 
 - DEAP
 - Tensorflow-datasets 
 - Scikit-learn
 - Pandas

## Example

The easiest example of this library (taken from <a href="https://github.com/IvanHCenalmor/deatf/blob/main/examples/simple.py">simple.py</a> in the examples folder), where every used parameter is predifined is the following one:

```
import numpy as np

from deatf.auxiliary_functions import load_fashion
from deatf.network import MLPDescriptor
from deatf.evolution import Evolving

from sklearn.preprocessing import OneHotEncoder

x_train, y_train, x_test, y_test, x_val, y_val = load_fashion()

OHEnc = OneHotEncoder()

y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()
y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
y_val = OHEnc.fit_transform(np.reshape(y_val, (-1, 1))).toarray()

e = Evolving(evaluation="XEntropy", desc_list=[MLPDescriptor], compl=False,
         x_trains=[x_train], y_trains=[y_train], x_tests=[x_val], y_tests=[y_val], 
         n_inputs=[[28, 28]], n_outputs=[[10]], batch_size=150, iters=10, 
         population=15, generations=10, max_num_layers=10, max_num_neurons=20,
         seed=0, dropout=False, batch_norm=False, evol_alg='mu_plus_lambda',
         evol_kwargs={'mu':10, 'lambda_':15, 'cxpb':0., "mutpb": 1.},
         sel = 'best')

a = e.evolve()
```
More complex cases can be created, an example of it (without being to complex) is <a href="https://github.com/IvanHCenalmor/deatf/blob/main/examples/sequential.py">sequential.py</a>.  In this second example, the evaluation function is not predefined; instead it is defined by the user in eval_sequential function. With this option, user can decide how to evaluate the created models.

```
import tensorflow as tf
import numpy as np

from deatf.auxiliary_functions import accuracy_error, load_fashion
from deatf.network import MLPDescriptor
from deatf.evolution import Evolving

from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as opt
from sklearn.preprocessing import OneHotEncoder

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]

def eval_sequential(nets, train_inputs, train_outputs, batch_size, iters, test_inputs, test_outputs, hypers):
   
    inp = Input(shape=train_inputs["i0"].shape[1:])
    out = Flatten()(inp)
    out = nets["n0"].building(out)
    out = nets["n1"].building(out)

    model = Model(inputs=inp, outputs=out)
    
    opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])
    
    model.compile(loss=tf.nn.softmax_cross_entropy_with_logits, optimizer=opt, metrics=[])
    
    model.fit(train_inputs['i0'], train_outputs['o0'], epochs=iters, batch_size=batch_size, verbose=0)

    pred = model.predict(test_inputs['i0'])
        
    res = tf.nn.softmax(pred)

    return accuracy_error(test_outputs["o0"], res),

x_train, y_train, x_test, y_test, x_val, y_val = load_fashion()

OHEnc = OneHotEncoder()

y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()
y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
y_val = OHEnc.fit_transform(np.reshape(y_val, (-1, 1))).toarray()

e = Evolving(evaluation=eval_sequential, desc_list=[MLPDescriptor, MLPDescriptor], 
             x_trains=[x_train], y_trains=[y_train], x_tests=[x_val], y_tests=[y_val], 
             batch_size=150, population=10, generations=10, iters=10, 
             n_inputs=[[28, 28], [10]], n_outputs=[[10], [10]], cxp=0.5, mtp=0.5, 
             hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]},
             batch_norm=False, dropout=False)
a = e.evolve()
```

## References
<a id="1">[1]</a> 
Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., ... & Ghemawat, S. (2016). Tensorflow: Large-scale machine learning on heterogeneous distributed systems. arXiv preprint arXiv:1603.04467.

<a id="2">[2]</a> 
Fortin, F. A., Rainville, F. M. D., Gardner, M. A., Parizeau, M., & Gagné, C. (2012). DEAP: Evolutionary algorithms made easy. Journal of Machine Learning Research, 13(Jul), 2171-2175.

<a id="3">[3]</a> 
Garciarena, U., Santana, R., & Mendiburu, A. (2018, July). Evolved GANs for generating Pareto set approximations. In Proceedings of the Genetic and Evolutionary Computation Conference (pp. 434-441). ACM.
