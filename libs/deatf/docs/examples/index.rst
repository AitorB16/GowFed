Examples
===============

This section show and explain all the examples used in DEATF. These examples are divided in two: the simple cases where predefined network descriptors are evaluated individualy and the custom cases that use custom evaluation functions and network descriptors are combined in order to create more complex networks.

Examples of the predefined networks
-----------------------------------
These cases are a collection of files where each one is for testing a predefined network. Due to that each file only uses one type of network, these files are ideal for testing. If new mutations are added or any change is done to the :py:class:`~deatf.network.Network` or :py:class:`~deatf.network.NetworkDescriptor` these would be the test to verify if everything has been correctly done.

.. toctree::
   :maxdepth: 4

   aux_functions_testing
   test_mlp
   test_cnn
   test_tcnn
   test_rnn

Custom examples
---------------
These other cases are examples of applications where combinations and more complex models are used. They also define different levels of complexity in the evaluation method: predefined evalautions, creating functions and even creating the step of training.   

.. toctree::
   :maxdepth: 4

   auto_encoder
   cnn_ae
   cnn_class
   gan
   multi
   rnn
   sequential
   simple
   single_skip
   wann
