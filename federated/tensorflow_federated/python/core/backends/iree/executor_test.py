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

import asyncio

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.iree import backend_info
from tensorflow_federated.python.core.backends.iree import executor
from tensorflow_federated.python.core.impl.context_stack import set_default_context
from tensorflow_federated.python.core.impl.execution_contexts import sync_execution_context
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation


class ExecutorTest(tf.test.TestCase):

  def test_float(self):
    ex = executor.IreeExecutor(backend_info.VULKAN_SPIRV)
    float_val = asyncio.run(ex.create_value(10.0, tf.float32))

    self.assertIsInstance(float_val, executor.IreeValue)
    self.assertEqual(str(float_val.type_signature), 'float32')
    self.assertIsInstance(float_val.internal_representation, np.float32)
    self.assertEqual(float_val.internal_representation, 10.0)
    result = asyncio.run(float_val.compute())
    self.assertEqual(result, 10.0)

  def test_constant_computation(self):
    ex = executor.IreeExecutor(backend_info.VULKAN_SPIRV)

    @tensorflow_computation.tf_computation
    def comp():
      return 1000.0

    comp_val = asyncio.run(ex.create_value(comp))

    self.assertIsInstance(comp_val, executor.IreeValue)
    self.assertEqual(str(comp_val.type_signature), '( -> float32)')
    self.assertTrue(callable(comp_val.internal_representation))
    # NOTE: The internal representation is a functions that takes a parameter
    # kwarg and returns a dict with a 'result' key.
    result = comp_val.internal_representation()['result']
    self.assertEqual(result, 1000.0)

    with self.assertRaises(TypeError):
      asyncio.run(comp_val.compute())

    result_val = asyncio.run(ex.create_call(comp_val))
    self.assertIsInstance(result_val, executor.IreeValue)
    self.assertEqual(str(result_val.type_signature), 'float32')
    self.assertIsInstance(result_val.internal_representation, np.float32)
    self.assertEqual(result_val.internal_representation, 1000.0)
    result = asyncio.run(result_val.compute())
    self.assertEqual(result, 1000.0)

  def test_add_one(self):
    ex = executor.IreeExecutor(backend_info.VULKAN_SPIRV)

    @tensorflow_computation.tf_computation(tf.float32)
    def comp(x):
      return x + 1.0

    comp_val = asyncio.run(ex.create_value(comp))

    self.assertEqual(str(comp_val.type_signature), '(float32 -> float32)')
    self.assertTrue(callable(comp_val.internal_representation))
    # NOTE: The internal representation is a functions that takes a parameter
    # kwarg and returns a dict with a 'result' key.
    result = comp_val.internal_representation(
        parameter=np.float32(5.0))['result']
    self.assertEqual(result, 6.0)

    arg_val = asyncio.run(ex.create_value(10.0, tf.float32))
    result_val = asyncio.run(ex.create_call(comp_val, arg_val))

    self.assertIsInstance(result_val, executor.IreeValue)
    self.assertEqual(str(result_val.type_signature), 'float32')
    result = asyncio.run(result_val.compute())
    self.assertEqual(result, 11.0)

  def test_as_default_context(self):
    ex = executor.IreeExecutor(backend_info.VULKAN_SPIRV)
    factory = executor_stacks.ResourceManagingExecutorFactory(
        executor_stack_fn=lambda _: ex)
    context = sync_execution_context.ExecutionContext(factory)
    set_default_context.set_default_context(context)

    @tensorflow_computation.tf_computation(tf.float32)
    def comp(x):
      return x + 1.0

    self.assertEqual(comp(10.0), 11.0)


if __name__ == '__main__':
  tf.test.main()
