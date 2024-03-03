from util.util import *
from logreg import LogisticRegression

def main():
    
    path = "data/water.train"
    X, T = loadDataset(path)
    T = toHotEncoding(T, 2)
    
    model = LogisticRegression()
    model.fit(X, T)
main() 