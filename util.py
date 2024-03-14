import numpy as np
from numpy.typing import NDArray

def softmax(a):
    ea = np.exp(a - np.max(a, axis=1, keepdims=True))
    return ea / np.sum(ea, axis=1, keepdims=True)

def accuracy(y_true, y_pred):
    return np.sum(np.argmax(y_true,axis=1) == np.argmax(y_pred, axis=1)) / y_true.shape[0]

def toHotEncoding(t, k):
    ans = np.zeros((t.shape[0], k))
    ans[np.arange(t.shape[0]), np.reshape(t, t.shape[0]).astype(int)] = 1
    return ans

def loadDataset(filename):
    """ Read CSV data from file.
    Returns X, Y values with hot targets. """
    labels=open(filename).readline().split(',')
    data=np.loadtxt(filename, delimiter=',')
    X=data[:,:-1] # observations
    T=data[:,-1]  # targets, discrete categorical features (integers)
    K=1 + np.max(T).astype(int) # number of categories
    N=X.shape[0]  # number of observations
    D=X.shape[1]  # number of features per observation
    return X, T

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