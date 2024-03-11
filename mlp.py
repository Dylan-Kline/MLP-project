from util.util import *
import numpy as np
from numpy.typing import NDArray

class MultilayerPerceptron:

    '''
        Makes use of the idea that each layer is itself a vector, to thus vectorize propagation of input.
        '''
    def __init__(self):
        self.num_hidden_layers = None
        self.learning_rate = 0.03
        self.iterations = 1000
        self.batch_size = 16

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