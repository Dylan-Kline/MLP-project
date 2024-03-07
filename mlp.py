from util.util import *
import numpy as np

class MultilayerPerceptron:

    def __init__(self):
        self.num_hidden_layers: None
        self.weights = None
        self.learning_rate = 0.03
        self.iterations = 1000
        self.batch_size = 16
