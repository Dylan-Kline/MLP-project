import numpy as np
from numpy.typing import NDArray

class NeuronLayer:

    def __init__(self, num_incoming_connections, num_neurons, activation_func, deriv_activation):
        self.weights = np.random.randn(num_neurons, num_incoming_connections) 
        self.activation_func = activation_func
        self.deriv_func = deriv_activation

    def forward(self, input: NDArray):
        self.input = input
        self.output = self.activation_func(np.dot(self.weights, self.input))
        return self.output
    
    def backward(self, output_error, learning_rate):

        # compute the error and gradient for the current layer 
        error = self.deriv_func(self.output) * output_error
        gradient = np.dot(error, self.input.T)