from util.util import *
import numpy as np
from numpy.typing import NDArray
from nn.NeuronLayer import NeuronLayer

class MultilayerPerceptron:

    '''
        Makes use of the idea that each layer is itself a vector, to thus vectorize propagation of input.
        '''
    def __init__(self):

        self.activation_functions = [NeuronLayer.tanh, softmax] # activation for each hidden layer and the output layer
        self.activation_derivatives = [NeuronLayer.tanh_derivative, ]
        self.layer_sizes = [61, 10, 1] # sizes for each layer from the input (index 0) to output layer (index n - 1)
        self.layers = list()
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(NeuronLayer(self.layer_sizes[i], self.layer_sizes[i+1], ))

        self.learning_rate = 0.03
        self.iterations = 1000
        self.batch_size = 16 # size of stochastic batches

    def fit(self, x: NDArray, y: NDArray):
        '''
            Trains the model on the given input dataset.
            @ x : numpy array of input data
            @ y : numpy array of one-hot encoding of the true outputs
            '''
        ''' Initalize the layer weights here since we cannot have arguments for the constructor '''
        num_samples, num_features = x.shape

        # for each iteration propagate the inputs and back prop the error
        for _ in range(self.iterations):
            
            # batch stochastic gradient descent
            for i in range(0, num_samples, self.batch_size):

                # create input and true output batchs
                x_batch = x[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                # propagate batches
                y_pred = self.propagate(x_batch)

                # compute errors and back prop
    
    def propagate(self, inputs: NDArray):
        pass

        #for unit in self.hidden_units