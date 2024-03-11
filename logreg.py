from util.util import *
import numpy as np

class LogisticRegression:

    def __init__(self):
        self.weights = None
        self.learning_rate = 0.03
        self.iterations = 1000
        self.batch_size = 16

    def initialize_weights(self, num_features, num_classes):
        self.weights = np.random.normal(0, 1, (num_features, num_classes)) # num features contains +1 for bias term

    def fit(self, data, targets):

        num_samples, num_features = data.shape
        num_classes = targets.shape[1]
        self.initialize_weights(num_features, num_classes)
        
        for _ in range(self.iterations):
            
            # randomize the indices of the data and select a random sample and its corresponding target
            index = np.random.permutation(num_samples)
            data = data[index]
            targets = targets[index]
            
            for i in range(0, num_samples, self.batch_size):
                
                # Grab the sample batches from dataset
                x_batch = data[i:i + self.batch_size]
                y_batch = targets[i:i + self.batch_size]
                
                # Compute the model's prediction and calculate error
                z = np.dot(x_batch, self.weights)
                output = softmax(z)
                error = output - y_batch
                
                # Gradient Calculations
                gradient = np.dot(x_batch.T, error)
                
                # update the weights
                self.weights -= self.learning_rate * gradient
  
                if _ % 100 == 0:
                    print(f"Current accuracy of the model: {accuracy(y_batch, output)} ")

                
    def predict(self, X):
        z = np.dot(X, self.weights)
        prediction = softmax(z)
        return prediction
     

