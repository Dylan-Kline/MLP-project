from util import *
from logreg import LogisticRegression
from mlp import MultilayerPerceptron

def main():
    
    path = "data/water.train"
    validation = "data/water.val"

    X, T = loadDataset(path)
    X_val, T_val = loadDataset(validation)

    T = toHotEncoding(T, 2)
    T_val = toHotEncoding(T_val, 2)

    # Train and evaluate the Logistic Regression Model
    model = LogisticRegression()
    model.fit(X, T)
    Y = model.predict(X)
    acc_train = accuracy(Y, T)
    print(acc_train)

    Y = model.predict(X_val)
    acc_train = accuracy(Y, T_val)
    print(acc_train)

    # Save logistic regression model
    model.save("logreg_new.model")

    # Train and evaluate the MLP Model
    mlp_model = MultilayerPerceptron()
    mlp_model.fit(X, T)
    Y = mlp_model.predict(X)
    acc_train2 = accuracy(T, Y)
    print(acc_train2)

    Y = mlp_model.predict(X_val)
    acc_train2 = accuracy(Y, T_val)

    # Save mlp model
    mlp_model.save("mlp_new.model")

    # Load saved mlp model
    loaded_mlp = MultilayerPerceptron.load("water.model")
    Y = loaded_mlp.predict(X_val)
    acc_train3 = accuracy(Y, T_val)
    print(acc_train3)

    # Load saved logreg model
    loaded_logreg = LogisticRegression.load("logreg_new.model")
    Y = loaded_logreg.predict(X_val)
    acc_train4 = accuracy(Y, T_val)
    print(acc_train4)
    
main() 