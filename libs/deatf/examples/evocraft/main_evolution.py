import sys
sys.path.append('../..')

import re
import tensorflow as tf
import numpy as np

from deatf.evolution import Evolving
from deatf.network import MLPDescriptor

from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as opt

from interactive_loss import interactive_loss
from symmetry_loss import total_symmetry, variance_pennalty

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]

#f = open('datos.txt', 'w')

def evaluate(probabilities):
    """
    Evaluation method for the networks during evolution process.
    
    :param probabilities: Data that will be shown, it contains the probability
                         distribution that describes the build.
                         
    :return: Value calculated with the symmetry of the received values.
    """
    probabilities = tf.math.sigmoid(probabilities)
    loss = total_symmetry(probabilities)    
    return loss

def eval_model(nets, train_inputs, train_outputs, batch_size, iters, test_inputs, test_outputs, hypers):
    """
    The model that will evaluate the data is a simple MLP. Data passed to those 
    models in this case will be noise, models will have to predict a probability
    distribution with the size of the bounds of the build. The evaluation will
    measure the symmetry that is in each created build. Once the model is evaluated
    an example of its prediction will be created in Minecraft.
    
    :param nets: Dictionary with the networks that will be used to build the 
                 final network and that represent the individuals to be 
                 evaluated in the genetic algorithm.
    :param train_inputs: Input data for training, this data will only be used to 
                         give it to the created networks and train them.
    :param train_outputs: Output data for training, it will be used to compare 
                          the returned values by the networks and see their performance.
    :param batch_size: Number of samples per batch are used during training process.
    :param iters: Number of iterations that each network will be trained.
    :param test_inputs: Input data for testing, this data will only be used to 
                        give it to the created networks and test them. It can not be used during
                        training in order to get a real feedback.
    :param test_outputs: Output data for testing, it will be used to compare 
                         the returned values by the networks and see their real performance.
    :param hypers: Hyperparameters that are being evolved and used in the process.
    :return: Symmetry loss obtained with the test data that evaluates the true
             performance of the network.
    """
    
    #f.write(nets['n0'].descriptor)
    
    inp = Input(shape=train_inputs['i0'].shape[1:])
    out = nets['n0'].building(inp)
    out = Reshape(bounds+[len(wanted_blocks)])(out)
    model = Model(inputs=inp, outputs=out)
    
    #model.summary()
    #model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def train_step(x_train):
        
        with tf.GradientTape() as tape:   
            probabilities = model(x_train, training=True)
            loss = evaluate(probabilities)
            
        gradients_of_generator = tape.gradient(loss, model.trainable_variables)  
        opt.apply_gradients(zip(gradients_of_generator, model.trainable_variables))
        return loss
    
    for epoch in range(iters):
        loss = train_step(train_inputs['i0'])
        if epoch % 100 == 0:
            print("Epoch {:03d}: Loss: {}".format(epoch, loss))
            #f.write("Epoch {:03d}: Loss: {}".format(epoch, loss))
        
    probabilities = model(test_inputs['i0'])
    print(probabilities)
    ev = evaluate(probabilities)

    index = [int(''.join(re.findall(r'\d', model.name))) if model.name != 'model' else 0, 0]    

    interactive_loss(probabilities, wanted_blocks, bounds, index, False)
    #f.write('Final ev: {}'.format(ev))

    return ev,


if __name__ == "__main__":
    
    bounds = [9,9,9]
    wanted_blocks = [-1,164, 169, 173]
    
    train_noise = np.random.normal(size=(5000, np.array(bounds).prod()))
    test_noise = np.random.normal(size=(2000,  np.array(bounds).prod()))
    val_noise = np.random.normal(size=(2000,  np.array(bounds).prod()))
    
    e = Evolving(desc_list=[MLPDescriptor], x_trains=[train_noise], y_trains=[train_noise], 
                 x_tests=[val_noise], y_tests=[val_noise], evaluation=eval_model, 
                 batch_size=200, population=5, generations=5, iters=500, 
                 n_inputs=[[train_noise.shape[1:]]], n_outputs=[bounds+[len(wanted_blocks)]], 
                 evol_alg='mu_comm_lambda', sel='best', 
                 cxp=0., mtp=1., hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]}, 
                 batch_norm=True, dropout=True)
    
    a = e.evolve()

    print(a[-1])
