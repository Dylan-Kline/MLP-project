from util.util import *
import numpy as np

class LogisticRegression:

    def __init__(self):
        self.weights = None
        self.learning_rate = 0.03
        self.iterations = 1000

    def initialize_weights(self, num_features, num_classes):
        self.weights = np.random.normal(0, 1, (num_features + 1, num_classes))

    def fit(data: NDArray, targets):

        num_samples, num_features = data.shape

