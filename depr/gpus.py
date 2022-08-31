import collections
import time

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8),
                         device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

gpu_devices = tf.config.list_physical_devices('GPU')
print(gpu_devices)

tf.config.set_logical_device_configuration(
    gpu_devices[0], 
    [tf.config.LogicalDeviceConfiguration(memory_limit=11000),
     tf.config.LogicalDeviceConfiguration(memory_limit=11000)])
tf.config.set_logical_device_configuration(
    gpu_devices[1], 
    [tf.config.LogicalDeviceConfiguration(memory_limit=11000),
     tf.config.LogicalDeviceConfiguration(memory_limit=11000)])
print(tf.config.list_logical_devices())

@tff.federated_computation
def hello_world():
  return 'Hello, World!'

print(hello_world())