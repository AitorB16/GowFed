Multi Layer Perceptron
======================

.. automodule:: simple
   :members:
   :undoc-members:
   :show-inheritance:

First of all fashion mnist dataset has to be loaded and preprocessed in order to
pass it to the network. In that preprocessing, labels that are integers from 0 to
9, are one hot encoded. That is, 3 turns into a vector [0,0,0,1,0,0,0,0,0,0] with 
a one in the index 3 (starting to count from 0) and the rest are zeros.

.. literalinclude:: /../examples/simple.py
   :lines: 23-29

Then is time to star the evolution. First specifying the desired parameters for evolution and then calling evolve function, evolution will be carried out.

.. literalinclude:: /../examples/simple.py
   :lines: 31-39   
