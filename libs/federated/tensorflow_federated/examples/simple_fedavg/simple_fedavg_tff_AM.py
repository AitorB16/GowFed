# Copyright 2020, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An implementation of the Federated Averaging algorithm.

This is intended to be a minimal stand-alone implementation of Federated
Averaging, suitable for branching as a starting point for algorithm
modifications; see `tff.learning.build_federated_averaging_process` for a
more full-featured implementation.

Based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

from pydoc import cli
from time import process_time_ns
import tensorflow as tf
import tensorflow_federated as tff

import sys

sys.path.append("/home/tester/Desktop/TF/federated/tensorflow_federated/examples/simple_fedavg")
print(sys.path)

from simple_fedavg_tf import build_server_broadcast_message
from simple_fedavg_tf import client_update
from simple_fedavg_tf import server_update
from simple_fedavg_tf import ServerState


def _initialize_optimizer_vars(model: tff.learning.Model,
                               optimizer: tf.keras.optimizers.Optimizer):
  """Creates optimizer variables to assign the optimizer's state."""
  # Create zero gradients to force an update that doesn't modify.
  # Force eagerly constructing the optimizer variables. Normally Keras lazily
  # creates the variables on first usage of the optimizer. Optimizers such as
  # Adam, Adagrad, or using momentum need to create a new set of variables shape
  # like the model weights.
  model_weights = tff.learning.ModelWeights.from_model(model)
  zero_gradient = [tf.zeros_like(t) for t in model_weights.trainable]
  optimizer.apply_gradients(zip(zero_gradient, model_weights.trainable))
  assert optimizer.variables()


def build_federated_averaging_process(
    model_fn,
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02)):
  """Builds the TFF computations for optimization using federated averaging.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for server update.
    client_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for client update.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  whimsy_model = model_fn()
  unfinalized_metrics_type = tff.framework.type_from_tensors(
      whimsy_model.report_local_unfinalized_metrics())
  metrics_aggregation_computation = tff.learning.metrics.sum_then_finalize(
      whimsy_model.metric_finalizers(), unfinalized_metrics_type)

  @tff.tf_computation
  def server_init_tf():
    model = model_fn()
    model_weights = tff.learning.ModelWeights.from_model(model)
    server_optimizer = server_optimizer_fn()
    _initialize_optimizer_vars(model, server_optimizer)
    return ServerState(
        model=model_weights,
        optimizer_state=server_optimizer.variables(),
        round_num=0)

  server_state_type = server_init_tf.type_signature.result
  model_weights_type = server_state_type.model

  @tff.tf_computation(server_state_type, model_weights_type.trainable)
  def server_update_fn(server_state, model_delta):
    model = model_fn()
    server_optimizer = server_optimizer_fn()
    _initialize_optimizer_vars(model, server_optimizer)
    return server_update(model, server_optimizer, server_state, model_delta)

  @tff.tf_computation(server_state_type)
  def server_message_fn(server_state):
    return build_server_broadcast_message(server_state)

  server_message_type = server_message_fn.type_signature.result
  tf_dataset_type = tff.SequenceType(whimsy_model.input_spec)

  @tff.tf_computation(tf_dataset_type, server_message_type)
  def client_update_fn(tf_dataset, server_message):
    model = model_fn()
    client_optimizer = client_optimizer_fn()
    return client_update(model, tf_dataset, server_message, client_optimizer)

  federated_server_state_type = tff.type_at_server(server_state_type)
  federated_dataset_type = tff.type_at_clients(tf_dataset_type)
 
  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type)
  def run_one_round1(server_state, federated_dataset):
    """Orchestration logic for one round of computation (part 1).

    Args:
      server_state: A `ServerState` containing the state of the training process
        up to the current round.
      federated_dataset: A federated `tf.data.Dataset` with placement
        `tff.CLIENTS` containing data to train the current round on.

    Returns:
      A tuple of updated `ServerState` and `client_outputs` corresponsing to local models' weights.
    """
    server_message = tff.federated_map(server_message_fn, server_state)
    server_message_at_client = tff.federated_broadcast(server_message)

    client_outputs = tff.federated_map(
        client_update_fn, (federated_dataset, server_message_at_client))

    return server_state, client_outputs.model_output, client_outputs.client_weight#, client_outputs.weights_delta
  
  
  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type, tff.FederatedType(tf.float32, tff.CLIENTS))#, tff.FederatedType(tff.SequenceType(tf.float32), tff.CLIENTS))
  def run_one_round2(server_state, federated_dataset, cli_weights):#, new_delta):
    """Orchestration logic for one round of computation (part 2).

    Args:
      server_state: A `ServerState` containing the state of the training process
        up to the current round.
      federated_dataset: A federated `tf.data.Dataset` with placement
        `tff.CLIENTS` containing data to train the current round on.

    Returns:
      A tuple of updated `ServerState` and `tf.Tensor` of average loss.
    """
    server_message = tff.federated_map(server_message_fn, server_state)
    server_message_at_client = tff.federated_broadcast(server_message)

    client_outputs = tff.federated_map(
        client_update_fn, (federated_dataset, server_message_at_client))

    #Weight is proportional to dataset size
    #weight_denom = client_outputs[0].client_weight
        
    #FEDAVG
    aggregated_outputs = metrics_aggregation_computation(
        client_outputs.model_output)

    #External AM weights
    weight_denom = cli_weights

    round_model_delta = tff.federated_mean(
        client_outputs.weights_delta, weight=weight_denom)

    server_state = tff.federated_map(server_update_fn,
                                     (server_state, round_model_delta))

    return server_state, aggregated_outputs

  @tff.federated_computation
  def server_init_tff():
    """Orchestration logic for server model initialization."""
    return tff.federated_eval(server_init_tf, tff.SERVER)

  return tff.templates.IterativeProcessAM(
      initialize_fn=server_init_tff, next_fn=run_one_round1, next_adv_fn=run_one_round2)
  
