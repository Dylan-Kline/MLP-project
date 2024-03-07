from util.util import *
from logreg import LogisticRegression
from matplotlib import pyplot as plt

def main():
    
    path = "data/water.train"
    validation = "data/water.val"
    X, T = loadDataset(path)
    X_val, T_val = loadDataset(validation)
    T = toHotEncoding(T, 2)
    T_val = toHotEncoding(T_val, 2)

    # Add bias term to inputs
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    
    model = LogisticRegression()
    model.fit(X, T)
    Y = model.predict(X)
    acc_train = accuracy(T, Y)
    print(acc_train)
    
main() 