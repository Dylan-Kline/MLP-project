from util.util import *
from logreg import LogisticRegression
from mlp import MultilayerPerceptron
from matplotlib import pyplot as plt

def main():
    
    path = "data/loans.train"
    validation = "data/loans.val"

    X, T = loadDataset(path)
    X_val, T_val = loadDataset(validation)

    T = toHotEncoding(T, 2)
    T_val = toHotEncoding(T_val, 2)

    X = normailze_data(X)

    # Train and evaluate the Logistic Regression Model
    model = LogisticRegression()
    model.fit(X, T)
    Y = model.predict(X)
    acc_train = accuracy(Y, T)
    print(acc_train)

    # Save logistic regression model
    model.save("logreg.model")

    # Train and evaluate the MLP Model
    mlp_model = MultilayerPerceptron()
    mlp_model.fit(X, T)
    Y = mlp_model.predict(X)
    acc_train2 = accuracy(T, Y)
    print(acc_train2)

    # Save mlp model
    mlp_model.save("mlp.model")

    # Load saved mlp model
    loaded_mlp = MultilayerPerceptron.load("mlp.model")
    Y = loaded_mlp.predict(X_val)
    acc_train3 = accuracy(Y, T_val)
    print(acc_train3)

    # Load saved logreg model
    loaded_logreg = LogisticRegression.load("logreg.model")
    Y = loaded_logreg.predict(X_val)
    acc_train4 = accuracy(Y, T_val)
    print(acc_train4)
    
main() 