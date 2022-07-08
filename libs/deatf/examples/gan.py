"""
This is a use case of DEATF where a Generative Adeversarial Network (GAN) is used.

The GAN is created by using two MLP. This is an unsupervised problem where two 
networks are used: one responsible of generating data (called generator) that 
is similar but not equal to the input data and the other one resposible to detect 
if the received data is real or created by the other network (called discriminator). 
The key in GANs is the interaction between those two networks. 
"""
import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np

from deatf.auxiliary_functions import batch, load_fashion
from deatf.network import MLPDescriptor
from deatf.evolution import Evolving

from tensorflow.keras.layers import Input, Flatten, Reshape
from tensorflow.keras.models import Model

def generator_loss(fake_out):
    """
    Loss function for the generator network. This function might seem complex, 
    in order to understand it the function of crossentropy has to be explained:
        
        cross_entropy = -( p(x) * log(q(x)) + (1 - p(x)) * log(1 - q(x)) )
        
    In it p(x) is the probability of the target and q(x) is the probability of 
    the prediction. If the probability of the target p(x) is 1, the second part
    of the equation would desapear. Otherwise, if p(x) is 0, the first part
    of the equation would be the one that desapears. The loss function of the 
    generator consist in reducing the mean of -log(q(x)).
    
    :param fake_out: Output of the generator model.
    :return: The mean of -log(fake_out).
    """
    
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_out,
                                                                    labels=tf.ones_like(fake_out)))
    return g_loss

def discriminator_loss(fake_out, real_out):
    """
    Loss function for the discriminator network. With the explanation of the 
    cross entropy from the genereator_loss function is easier to understand
    the discriminator loss function. At the end, it is:
        
        d_loss = -log(real_out) - log(1 - fake_out) 
    
    :param fake_out: Output of the generator model.
    :param real_out: Real data, features given to the GAN.
    :return: - (log(real_out) + log(1 - fake_out))
    """
    
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_out, 
                                                                         labels=tf.ones_like(real_out)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_out,
                                                                         labels=tf.zeros_like(fake_out)))
    
    d_loss = d_loss_real + d_loss_fake
    return d_loss

def eval_gan(nets, train_inputs, _, batch_size, iters, __, ___ ,____):
    """
    This case is more complex than other examples, in it two models are used
    the generator and the discriminator. They are two separete models, but they 
    are related by their loss function. The generator is meant to create outputs 
    that are similar but not equal to the real input and the discriminator
    has to diferentiate between the reals and the generated by the generator.
    That is why their losses are linked and the train step is more complex and
    tf.function ahs to be used to determine how the training has to go on.
    
    
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
    :return: Generator's loss function that evaluates the true
             performance of the network.
    """
      
    noise = np.random.normal(size=(150, 10))
    
    g_inp = Input(shape=noise.shape[1:])
    g_out = nets["n1"].building(g_inp)
    g_out = Reshape(x_train.shape[1:])(g_out)
    
    g_model = Model(inputs=g_inp, outputs=g_out)

    d_inp = Input(shape=x_train.shape[1:])
    d_out = Flatten()(d_inp)
    d_out = nets["n0"].building(d_out)
    
    d_model = Model(inputs=d_inp, outputs=d_out)
    
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def train_step(x_train):
        
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
    
            generated_images = g_model(noise, training=True)
            fake_out = d_model(generated_images, training=True)
            
            real_out = d_model(x_train, training=True)
            
            g_loss = generator_loss(fake_out)
            d_loss = discriminator_loss(fake_out, real_out)
            
        gradients_of_generator = g_tape.gradient(g_loss, g_model.trainable_variables)
        gradients_of_discriminator = d_tape.gradient(d_loss, d_model.trainable_variables)
    
        generator_optimizer.apply_gradients(zip(gradients_of_generator, g_model.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, d_model.trainable_variables))
    
    aux_ind = 0        
    
    for epoch in range(iters):

        image_batch = batch(train_inputs["i0"], batch_size, aux_ind)
        aux_ind = (aux_ind + batch_size) % train_inputs["i0"].shape[0]
        train_step(image_batch)
    
    noise = np.random.normal(size=(150, 10))
    
    generated_images = g_model(noise, training=False)
    
    return generator_loss(generated_images).numpy(),
    
if __name__ == "__main__":

    x_train, _, x_test, _, x_val, _ = load_fashion()

    # The GAN evolutive process is a common 2-DNN evolution

    e = Evolving(evaluation=eval_gan, desc_list=[MLPDescriptor, MLPDescriptor],
                 x_trains=[x_train], y_trains=[x_train], x_tests=[x_val], y_tests=[x_val], 
                 n_inputs=[[28, 28], [10]], n_outputs=[[1], [784]], 
                 population=5, generations=5, batch_size=150, iters=50, 
                 lrate=0.1, cxp=0.5, mtp=0.5, seed=0,
                 max_num_layers=10, max_num_neurons=100, max_filter=4, max_stride=3,
                 evol_alg='mu_plus_lambda', sel='best', sel_kwargs={}, 
                 hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]}, 
                 batch_norm=True, dropout=True)

    res = e.evolve()

    print(res[0])