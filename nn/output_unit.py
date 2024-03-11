from util.util import *
import numpy as np
from numpy.typing import NDArray

class OutputUnit:

    def __init__(self):
        self.incoming_weights = None # NDArray of weights incoming from the last hidden layer
        self.output = None
        self.delta_error = None

    def compute_output(self, input: NDArray):
        '''
            Computes the output of this output unit, which is computed using the softmax function.
            @ input : input data, numpy array
            return : softmax of input and weights going to this unit
            '''
        return softmax(np.dot(self.incoming_weights, input))