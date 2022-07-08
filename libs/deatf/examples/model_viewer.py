
import sys
sys.path.append('..')

import pandas as pd
import numpy as np

from deatf.auxiliary_functions import load_fashion
from deatf.network import MLPDescriptor
from deatf.evolution import Evolving

import tensorflow as tf

model = tf.keras.models.load_model("model.h5")
model.summary()

#print('done')