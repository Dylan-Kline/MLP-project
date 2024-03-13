from util.util import *
import numpy as np
import pickle

class LogisticRegression:

    def __init__(self):
        self.weights = None
        self.learning_rate = 0.03
        self.iterations = 100000
        self.batch_size = 128

    def initialize_weights(self, num_features, num_classes):
        self.weights = np.random.normal(0, 1, (num_features, num_classes)) # num features contains +1 for bias term

    def fit(self, data, targets):

        # add bias term to input data
        data = np.hstack([data, np.ones((data.shape[0], 1))])

        num_samples, num_features = data.shape
        num_classes = targets.shape[1]
        self.initialize_weights(num_features, num_classes)
        
        for _ in range(self.iterations):
            
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
            
            # update the weights
            self.weights -= self.learning_rate * gradient

            y_pred = softmax(np.dot(data, self.weights))
            if accuracy(targets, y_pred) > 0.65:
                break
  
            # if _ % 100 == 0:
            #     output = softmax(np.dot(data, self.weights))
            #     print(f"Current accuracy of the model: {accuracy(targets, output)} ")

                
    def predict(self, X):

        # add bias term to input data
        X = np.hstack([X, np.ones((X.shape[0], 1))])

        z = np.dot(X, self.weights)
        prediction = softmax(z)
        return prediction
    
    def save(self, path):
        '''
            Writes this logreg model to a file.
            '''
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        '''
        Load a logreg model from a file using pickle.
        '''
        with open(filename, 'rb') as file:
            return pickle.load(file)
     

