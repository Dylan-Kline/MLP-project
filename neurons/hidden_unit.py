import numpy as np
from numpy.typing import NDArray

class HiddenUnit:
    '''
        Hidden layer neuron units that have a tanh function output.
        '''

    def __init__(self):
        self.incoming_connections: None
        self.output: None
        
    def compute_output(self, input: NDArray):
        """
            Computes the output of the neuron unit using its incoming connections and the given input numpy array.
            Returns the output from a tanh function on the input array.
            """