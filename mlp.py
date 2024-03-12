from util.util import *
import numpy as np
from numpy.typing import NDArray
from nn.NeuronLayer import NeuronLayer
from nn.OutputLayer import OutputLayer

class MultilayerPerceptron:

    '''
        Makes use of the idea that each layer is itself a vector, to thus vectorize propagation of input.
        '''
    def __init__(self):

        # Hyperparameters for model
        self.learning_rate = 0.3
        self.iterations = 1000
        self.batch_size = 16 # size of stochastic batches

        # Parameters for neural network layers
        self.activation_functions = [NeuronLayer.tanh] # activation for each hidden layer and the output layer
        self.activation_derivatives = [NeuronLayer.tanh_derivative, NeuronLayer.tanh_derivative]


    def initialize_mlp(self, num_features: int):

        self.layer_sizes = [num_features, 10, 2] # sizes for each layer from the input (index 0) to output layer (index n - 1)

        # Creates the layers of the neural network
        self.layers = list()
        for i in range(len(self.layer_sizes) - 2):
            self.layers.append(NeuronLayer(self.layer_sizes[i], self.layer_sizes[i+1], 
                                           self.activation_functions[i], self.activation_derivatives[i]))
            
        # Add output layer to layers list
        self.layers.append(OutputLayer(self.layer_sizes[-2], self.layer_sizes[-1], softmax, self.activation_derivatives[1]))

    def fit(self, x: NDArray, y: NDArray):
        '''
            Trains the model on the given input dataset.
            @ x : numpy array of input data
            @ y : numpy array of one-hot encoding of the true outputs
            '''
        
        num_samples, num_features = x.shape # rows and columns of the input data x, respectively
        
        # initialize model weights
        self.initialize_mlp(num_features)

        # for each iteration propagate the inputs and back prop the error
        for _ in range(self.iterations):

            # shuffle the input data for stochastic gradient descent
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            
            # perform batch stochastic gradient descent
            for i in range(0, num_samples, self.batch_size):

                # create input and true output batchs
                x_batch = x_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                # propagate batches
                y_pred = self.predict(x_batch)

                # perform back propagation
                self.backprop(y_pred, y_batch)

                if _ % 100 == 0:
                    print(f"Current accuracy of the model: {accuracy(y_batch, y_pred)} ")
                
    
    def predict(self, inputs: NDArray):
        '''
            Performs forward propagation with the given input data.
            @ inputs : inputs data array
            return : prediction of model from forward pass of input
            '''
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backprop(self, predictions: NDArray, true_labels: NDArray):
        '''
            Performs backpropagation on the current network model.
            @ output_error : numpy array of delta errors
            '''
        
        # compute output error
        error = self.layers[-1].backward(predictions, true_labels, self.learning_rate)

        # Compute delta error for each hidden layer and update weights
        for layer in reversed(self.layers[:-1]):
            error = layer.backward(error, self.learning_rate)