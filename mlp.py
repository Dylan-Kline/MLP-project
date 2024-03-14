from util import *
import numpy as np
from numpy.typing import NDArray
from nn.NeuronLayer import NeuronLayer
from nn.OutputLayer import OutputLayer
import pickle

class MultilayerPerceptron:

    '''
        Makes use of the idea that each layer is itself a vector, to thus vectorize propagation of input.
        '''
    def __init__(self):

        # Hyperparameters for model
        self.learning_rate = 0.0003
        self.iterations = 10000
        self.batch_size = 256 # size of stochastic batches
        self.decay_rate = 0.00000002 # controls the rate of learning rate decay
        self.lambda_reg = .001

        # Parameters for neural network layers
        self.activation_functions = list() # activation for each hidden layer and the output layer
        self.activation_derivatives = list()
        self.layer_sizes = None
        self.layers = None

    def initialize_mlp(self, num_features: int):
        '''
            Initializes the layers of the multilayer perceptron.
            @ num_features : number of features/attributes in the input data'''
        
        if self.layer_sizes is None:
            self.layer_sizes = self.layer_sizes = [num_features, 10, 32, 16, 8, 4, 2] # sizes for each layer from the input (index 0) to output layer (index n - 1)

            # init activation functions and derivatives to be used for each layer
            for i in range(len(self.layer_sizes) - 1):
                # hidden layers
                self.activation_functions.append(NeuronLayer.tanh) 
                self.activation_derivatives.append(NeuronLayer.tanh_derivative)

            # output layer
            self.activation_functions.append(softmax) 
            self.activation_derivatives.append(NeuronLayer.tanh_derivative)

        # Creates the layers of the neural network
        self.layers = list()
        for i in range(len(self.layer_sizes) - 2):
            self.layers.append(NeuronLayer(self.layer_sizes[i] + 1, self.layer_sizes[i+1], 
                                           self.activation_functions[i], self.activation_derivatives[i]))
            
        # Add output layer to layers list
        self.layers.append(OutputLayer(self.layer_sizes[-2] + 1, self.layer_sizes[-1], softmax, self.activation_derivatives[1]))

    def fit(self, x: NDArray, y: NDArray):
        '''
            Trains the model on the given input dataset.
            @ x : numpy array of input data
            @ y : numpy array of one-hot encoding of the true outputs
            '''

        # normalize the input data
        x = normailze_data(x)

        # Grab dimensions of input data
        num_samples, num_features = x.shape # rows and columns of the input data x, respectively
        
        # initialize model weights
        self.initialize_mlp(num_features)

        if num_samples < self.batch_size:
                self.batch_size = num_samples // 6

        # for each iteration propagate the inputs and back prop the error
        for l in range(self.iterations):

            # Randomly sample a batch from the dataset
            indices = np.random.choice(num_samples, self.batch_size, replace=False)
            x_batch = x[indices]
            y_batch = y[indices]

            # propagate batches
            y_pred = self.predict_normalized(x_batch)
            
            # perform back propagation
            self.backprop(y_pred, y_batch)

            if l % 1000 == 0:
                y_pred = self.predict_normalized(x)
                print(f"Current accuracy of the model: {accuracy(y, y_pred)} ")

            # Decay the learning rate as iterations progress
            self.learning_rate = self.learning_rate / (1.0 + self.decay_rate * float(l))
                
    
    def predict_normalized(self, inputs: NDArray):
        '''
            Performs forward propagation with the given input data.
            @ inputs : normalized input data array
            return : prediction of model from forward pass of input
            '''
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
            #print(output)
        return output
    
    def predict(self, input: NDArray):
        '''
            Performs forward propagation with the given input data.
            @ inputs : unnormalized input data array
            return : prediction of model from forward pass of input
            '''
        input = normailze_data(input)

        output = input
        for layer in self.layers:
            output = layer.forward(output)
            #print(output)
        return output
    
    def backprop(self, predictions: NDArray, true_labels: NDArray):
        '''
            Performs backpropagation on the current network model.
            @ output_error : numpy array of delta errors
            '''
        
        # compute output error
        error = self.layers[-1].backward(predictions, true_labels, self.learning_rate, self.lambda_reg)

        # Compute delta error for each hidden layer and update weights
        for layer in reversed(self.layers[:-1]):
            error = layer.backward(error, self.learning_rate, self.lambda_reg)

    def save(self, path):
        '''
            Writes this mlp model to a file.
            '''
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        '''
        Load a model from a file using pickle.
        '''
        with open(filename, 'rb') as file:
            return pickle.load(file)
    
    def print_weights(self):
        '''
        Outputs the model's weights to console.
        '''
        for layer in self.layers:
            print(layer.get_weights())