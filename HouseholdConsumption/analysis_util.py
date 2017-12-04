import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_dataset(path, na_values = np.NaN):
    return pd.read_csv(path, low_memory=False, na_values=na_values)


def calc_regression(regressor, name):
    dataset = load_dataset("../datasets/householdConsumption/household_power_consumption_preprocessed.txt")
    dependent_variable = 'Global_intensity'
    dataset_short_name='HEPC'

    # we need neither time nor date
    dataset = dataset.drop('Time', axis = 1)
    dataset = dataset.drop('Date', axis = 1)

    X = dataset.drop(dependent_variable, axis = 1)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X , dataset[dependent_variable], test_size = 0.3, random_state = 5
    )

    # training
    regressor.fit(X, dataset[dependent_variable])
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
    plt.xlabel("Actual global intensity")
    plt.ylabel("Predicted global intensity")
    plt.title("Actual vs Predicted global intensity")
    plt.savefig("images/" + name + "-actualVsPredict" + dataset_short_name + ".png")

    # residual plot
    plt.cla()
    plt.scatter(pred_train, pred_train - Y_train, c='b', s=1, alpha=0.5)
    plt.scatter(pred_test, pred_test - Y_test, c = 'g', s=1)
    plt.hlines(y=0, xmin=0, xmax = 1)
    plt.title('Residual Plot using training (blue) and test (green) data')
    plt.ylabel('Residuals')
    plt.savefig("images/" + name + "-residuals" + dataset_short_name + ".png")

    print("Fit a model on X_train and calculate Mean-squared error with Y_train: ", np.mean((Y_train-pred_train) ** 2))
    print("Fit a model on X_train and calculate Mean-squared error with X_test, Y_test: ", np.mean((Y_test-pred_test) ** 2))
