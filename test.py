import numpy as np
from util.util import *

input = np.array([[1.5, 2.1, 1.0]])
weights = np.array([[0.7721859,  -1.48420843, -0.51994132],
                    [-0.43080562,  1.02945506, -0.78440931]])

print(input.shape)
print(weights.shape)

input = np.tanh(np.dot(input, weights.T))
print(input)

input = np.hstack([input, np.ones((input.shape[0], 1))])
weights = np.array([[0.02971024, -0.03101737,  0.65698973],
                    [0.29227427, -0.20128359, -0.0131016]])

print(input.shape)
print(weights.shape)
output = softmax(np.dot(input, weights.T))
print(output)