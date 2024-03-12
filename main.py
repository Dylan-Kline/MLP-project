from util.util import *
from logreg import LogisticRegression
from mlp import MultilayerPerceptron
from matplotlib import pyplot as plt

def main():
    
    path = "data/loans.train"
    validation = "data/water.val"

    X, T = loadDataset(path)
    X_val, T_val = loadDataset(validation)

    T = toHotEncoding(T, 2)
    T_val = toHotEncoding(T_val, 2)

    X = normailze_data(X)


    # X = np.array([1.5, 2.1])
    # # Reshape X to be a 2D array with 1 row and 3 columns
    # X = X.reshape(1, -1)
    # T = np.array([1.0])

    # print(X.shape)

    # T = toHotEncoding(T, 2)
    # print(T)
    
    # model = LogisticRegression()
    # model.fit(X, T)
    # Y = model.predict(X)
    # acc_train = accuracy(T, Y)
    # print(acc_train)

    mlp_model = MultilayerPerceptron()
    mlp_model.fit(X, T)
    Y = mlp_model.predict(X)
    acc_train2 = accuracy(T, Y)
    print(acc_train2)
    
main() 