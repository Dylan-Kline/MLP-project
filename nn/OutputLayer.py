from util.util import *
import numpy as np
from nn.NeuronLayer import NeuronLayer
from numpy.typing import NDArray

class OutputLayer(NeuronLayer):

    def backward(self, predictions: NDArray, true_labels: NDArray, learning_rate: float):
        '''
            Performs delta error backpropagation for the current layer.
            @ output_error : errors from the following layer
            @ learning_rate: learning rate of the model
            return : array of errors for previous layer input
            '''

        # compute the error and gradient for the current layer 
        output_error = predictions - true_labels
        gradient = np.dot(output_error.T, self.input)

        # input error
        input_error = np.dot(output_error, self.weights)

        # update the weights for this layer
        self.weights -= learning_rate * gradient

        # calculate input error for previous layer
        return input_error[:, :-1]