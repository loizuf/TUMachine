import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split


def load_dataset(path, na_values = np.NaN):
    return pd.read_csv(path, low_memory=False, na_values=na_values)


def draw_correlation():
    dataset = load_dataset("../datasets/electricity_load/LD2011_2014_preprocessed.txt")

    # correlation
    plt.scatter(dataset['client1'], dataset['client2'], s=2)
    plt.xlabel("Client 1 electricity load")
    plt.ylabel("Client 2 electricity load")
    plt.title("Positive correlation between two clients")
    plt.savefig("correlation.png")
    plt.clf()


def calc_regression(regressor, name):
    dataset = load_dataset("../datasets/electricity_load/LD2011_2014_preprocessed.txt")

    # we don't need time - it's irrelevant
    dataset = dataset.drop('Date', axis = 1)
    X = dataset.drop('client2', axis = 1)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X , dataset["client2"], test_size = 0.3, random_state = 5
    )

    # training
    regressor.fit(X, dataset["client2"])
    pred_train = regressor.predict(X_train)
    pred_test = regressor.predict(X_test)

    #####################
    #  results          #
    #####################
    fig, ax = plt.subplots()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid()
    plt.scatter(Y_test, pred_test, s=1)
    plt.xlabel("Actual electricity load")
    plt.ylabel("Predicted electricity load")
    plt.title("Actual vs Predicted electricity load")
    plt.savefig(name + "-actualVsPredict.png")

    # residual plot
    plt.cla()
    plt.scatter(pred_train, pred_train - Y_train, c='b', s=1, alpha=0.5)
    plt.scatter(pred_test, pred_test - Y_test, c = 'g', s=1)
    plt.hlines(y=0, xmin=0, xmax = 1)
    plt.title('Residual Plot using training (blue) and test (green) data')
    plt.ylabel('Residuals')
    plt.savefig(name + "-residuals.png")

    print("Fit a model on X_train and calculate Mean-squared error with Y_train: ", np.mean((Y_train-pred_train) ** 2))
    print("Fit a model on X_train and calculate Mean-squared error with X_test, Y_test: ", np.mean((Y_test-pred_test) ** 2))


if __name__ == '__main__':
    draw_correlation()
