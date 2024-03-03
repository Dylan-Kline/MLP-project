from util.util import *
from logreg import LogisticRegression

def main():
    
    path = "data/loans.train"
    X, T = loadDataset(path)
    T = toHotEncoding(T, 2)
    
    # Add bias term to inputs
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    
    model = LogisticRegression()
    model.fit(X, T)
    Y = model.predict(X)
    acc_train1 = accuracy(Y, T)
    print(acc_train1)
    
main() 