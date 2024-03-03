from util.util import *
import numpy as np

class LogisticRegression:

    def __init__(self):
        self.weights = None
        self.learning_rate = 0.03
        self.iterations = 1000

    def initialize_weights(self, num_features, num_classes):
        self.weights = np.random.normal(0, 1, (num_features, num_classes))

    def fit(self, data, targets):

        num_samples, num_features = data.shape
        num_classes = targets.shape[1]
        self.initialize_weights(num_features, num_classes)
        
        for _ in range(self.iterations):
            
            # randomize the indices of the data and select a random sample and its corresponding target
            index = np.random.randint(num_samples)
            x_sample = data[index]
            y_sample = targets[index]
            
            # Compute the model's prediction and calculate error
            z = np.dot(data, self.weights)
            break
                

