from util.util import *
from logreg import LogisticRegression

def main(path):
    
    X, T = loadDataset(path)
    T = toHotEncoding(T, 2)
    model = LogisticRegression()
main() 