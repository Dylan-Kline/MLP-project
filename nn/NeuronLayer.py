from util.util import *
import numpy as np
from numpy.typing import NDArray

class NeuronLayer:

    def __init__(self, num_incoming_connections, num_neurons, activation_func, deriv_activation=None):
        self.weights = np.random.randn(num_neurons, num_incoming_connections) # num units in layer X num weights to layer
        self.activation_func = activation_func
        self.deriv_func = deriv_activation

    def forward(self, input: NDArray):
        '''
            Performs feed-forward propagation with the given input
            @input : numpy array of input data
            return : output from the model based on the given input
            '''
        self.input = input
        self.output = self.activation_func(np.dot(self.input, self.weights.T))
        return self.output
    
    def backward(self, output_error: NDArray, learning_rate: float):
        '''
            Performs delta error backpropagation for the current layer.
            @ output_error : errors from the following layer
            @ learning_rate: learning rate of the model
            return : array of errors for previous layer input
            '''

        # compute the error and gradient for the current layer 
        error = self.deriv_func(self.output) * output_error
        gradient = np.dot(error.T, self.input)

        # update the weights for this layer
        self.weights -= learning_rate * gradient

        # calculate input error for previous layer
        input_error = np.dot(error, self.weights)
        return input_error
    
    @staticmethod
    def tanh(z: NDArray):
        '''
            Computes the tanh function on a numpy array z.
            @ z : numpy array from the dot product of weights and input data
            @ return : output from tanh function
            '''
        #return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        return np.tanh(z)
    
    @staticmethod
    def tanh_derivative(z: NDArray):
        '''
            Compute the tanh derivative on the given input array z.
            @ z : numpy array
            return : output from tanh prime
            '''
        
        return 1 - NeuronLayer.tanh(z) ** 2
    
    def print_weights(self):
        print(self.weights.shape)
        print(self.weights)