import numpy as np
from utils import *
from mlp import MultilayerPerceptron
from logreg import LogisticRegression

def main():
    path = "data/loans.train"
    validation = "data/loans.val"

    X, T = loadDataset(path)
    X_val, T_val = loadDataset(validation)

    T = toHotEncoding(T, 2)
    T_val = toHotEncoding(T_val, 2)

    mlp_model = MultilayerPerceptron.load_pickle("loans_711_728.model")
    mlp_model.fit(X, T)
    mlp_model.plot_accuracy_vs_iteration()

main()