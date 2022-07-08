import sys
sys.path.append('../..')

from deatf.network import MLPDescriptor, MLP

import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Model
from vectors_to_blocks import *

from interactive_loss import interactive_loss
from symmetry_loss import total_symmetry

def evaluate(probabilities):
    probabilities = tf.math.sigmoid(probabilities)
    loss = total_symmetry(probabilities[0])    
    return loss

def model_eval(network, train_data, test_data, training_iterations, bounds, wanted_blocks):    
    
    inp = Input(shape=train_data.shape[1:])
    out = network.building(inp)
    out = Reshape(bounds+[len(wanted_blocks)])(out)
    model = Model(inputs=inp, outputs=out)
    # model.summary()
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def train_step(x_train):
        
        with tf.GradientTape() as tape:   
            probabilities = model(x_train, training=True)
            loss = evaluate(probabilities)
            
        gradients_of_generator = tape.gradient(loss, model.trainable_variables)  
        opt.apply_gradients(zip(gradients_of_generator, model.trainable_variables))
        return loss
    
    for epoch in range(training_iterations):
        loss = train_step(train_data)     
        if epoch % 100 == 0:
            print("Epoch {:03d}: Loss: {}".format(epoch, loss))
        
    prediction = model.predict(test_data)
    ev = evaluate(prediction)

    index = [int(''.join(re.findall(r'\d', model.name))) if model.name != 'model' else 0, 0]    

    interactive_loss(prediction, wanted_blocks, bounds, index, False)

    return ev

if __name__ == '__main__':
    
    print('ADVISE: Remember to activate the server in order to see and evaluate the created structure after training.')
    
    bounds = [9,9]
    wanted_blocks = [-1,164, 169, 173]
    
    train_noise = np.random.normal(size=(5000, np.array(bounds).prod()))
    test_noise = np.random.normal(size=(2000,  np.array(bounds).prod()))
    val_noise = np.random.normal(size=(2000,  np.array(bounds).prod()))

    desc = MLPDescriptor(number_hidden_layers=10, input_dim=3, output_dim=len(wanted_blocks))
    desc.random_init(input_size=3, output_size=bounds+[len(wanted_blocks)], 
                    max_num_layers=10, max_num_neurons=200, max_stride=None, max_filter=None,
                    dropout=True, batch_norm=True)
    network = MLP(desc)

    iterations = [100, 500, 1000, 5000, 10000, 50000]

    for i in iterations:
        preds = model_eval(network, train_noise, test_noise, i, bounds, wanted_blocks)
        print(preds)

