from util import *
import numpy as np
from numpy.typing import NDArray
import json

class LogisticRegression:

    def __init__(self):
        self.weights = None
        self.learning_rate = 0.0003
        self.iterations = 20000
        self.batch_size = 1500
        self.lambda_reg = 0.0

        self.plot_data = list()

    def initialize_weights(self, num_features, num_classes):
        self.weights = np.random.normal(0, 1, (num_features, num_classes)) # num features contains +1 for bias term

    def fit(self, data, targets):

        # Normalize the data
        data = LogisticRegression.normailze_data(data)

        # add bias term to input data
        data = np.hstack([data, np.ones((data.shape[0], 1))])

        num_samples, num_features = data.shape
        num_classes = targets.shape[1]
        self.initialize_weights(num_features, num_classes)

        if num_samples < self.batch_size:
                self.batch_size = num_samples // 6
        
        for i in range(self.iterations):
            
            # Randomly sample a batch from the dataset
            indices = np.random.choice(num_samples, self.batch_size, replace=False)
            x_batch = data[indices]
            y_batch = targets[indices]
                
            # Compute the model's prediction and calculate error
            z = np.dot(x_batch, self.weights)
            output = softmax(z)
            error = output - y_batch
            
            # Gradient Calculations
            gradient = np.dot(x_batch.T, error)

            # Compute regularization term
            reg_term = self.lambda_reg * self.weights
            reg_term[:, -1] = 0 # exclude bias from regularization
            
            # update the weights
            self.weights -= self.learning_rate * (gradient + reg_term)

            y_pred = softmax(np.dot(data, self.weights))
            # if accuracy(targets, y_pred) > 0.62:
            #     break
  
            # if _ % 10000 == 0:
            #     output = softmax(np.dot(data, self.weights))
            #     print(f"Current accuracy of the model: {accuracy(targets, output)} ")

                
    def predict(self, X):

        # normalize the data
        X = LogisticRegression.normailze_data(X)

        # add bias term to input data
        X = np.hstack([X, np.ones((X.shape[0], 1))])

        z = np.dot(X, self.weights)
        prediction = softmax(z)
        return prediction
    
    def save(self, path):
        '''
            Writes this logreg model to a text file.
            '''
        model_data = {
            "learning rate": self.learning_rate,
            "iterations": self.iterations,
            "batch size": self.batch_size,
            "lambda": self.lambda_reg,
            "layer weights": self.get_weights().tolist()
        }

        with open(path, 'w') as file:
            json.dump(model_data, file)

    @staticmethod
    def load(path):
        '''
            Loads the a logreg model from a json text file
            '''
        with open(path, 'r') as file:
            model_data = json.load(file)
        
        # create new log reg model
        new_logmodel = LogisticRegression()

        # set model parameters based on loaded model data
        new_logmodel.weights = np.array(model_data['layer weights'])
        new_logmodel.learning_rate = model_data['learning rate']
        new_logmodel.batch_size = model_data['batch size']
        new_logmodel.lambda_reg = model_data['lambda']
        new_logmodel.iterations = model_data['iterations']

        return new_logmodel


 
    @staticmethod
    def normailze_data(data: NDArray):
        """
        Perform min-max normalization on a dataset.
        Each feature (column) is scaled to the range [0, 1] using the formula:
        (x - min) / (max - min)

        @ data: numpy array of data to be normalized
        return : normalized data
        """

        min_values = data.min(axis=0)
        max_values = data.max(axis=0)

        # Avoid division by zero in case max and min values are the same
        range_values = max_values - min_values
        range_values[range_values == 0] = 1

        # Apply the min-max normalization
        normalized_data = (data - min_values) / range_values
        return normalized_data
    
    def get_weights(self):
        return self.weights
    

