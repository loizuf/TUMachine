import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  cross_val_predict
from math import sqrt
import csv

def load_dataset(path):
    return pd.read_csv(path, low_memory=False)


def write_to_file(ids, predictions, method_name):
    subm = "outputs/studentPerform.predictions_" + method_name + ".csv"
    file_subm = open(subm, "w")
    pamwriter = csv.writer(file_subm)

    pamwriter.writerow(["id", "Grade"])
    for index in range(len(predictions)):
        # print([ids[index],predictions[index]])
        pamwriter.writerow([ids[index], predictions[index]])
    file_subm.close()

def calc_regression(regressor, name):
    dataset = load_dataset("../datasets/Student_Performance/StudentPerformance.shuf.train_preprocessedSTD.csv")

    dataset_test = load_dataset("../datasets/Student_Performance/StudentPerformance.shuf.test_preprocessedSTD.csv")
    dependent_variable = 'Grade'
    dataset_short_name='SP'

    # we dont need id
    dataset = dataset.drop('id', axis = 1)

    X = dataset.drop(dependent_variable, axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, dataset[dependent_variable], test_size=0.3, random_state=5
    )
    y_pred = cross_val_predict(regressor, dataset.drop(dependent_variable, axis=1), dataset[dependent_variable], cv=10)

    # training
    regressor.fit(X, dataset[dependent_variable])
    pred_train = regressor.predict(X_train)
    pred_test = regressor.predict(X_test)

    pred_test = np.round(pred_test,0)
    pred_train = np.round(pred_train,0)

    # pred_test = pred_test.astype(int)
    # pred_train = pred_train.astype(int)
    #####################
    #  results          #
    #####################
    fig, ax = plt.subplots()
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 20])
    ax.grid()
    plt.scatter(Y_test, pred_train)
    plt.xlabel("Actual" + dependent_variable)
    plt.ylabel("Predicted "+dependent_variable)
    plt.title("Actual vs Predicted "+ dependent_variable)
    plt.savefig("images/" + name + "-actualVsPredict" + dataset_short_name + ".png")

    # residual plot
    plt.cla()
    plt.scatter(pred_train, pred_train - Y_train, c='b', alpha=0.5)
    plt.scatter(pred_test, pred_test - Y_test, c='g')
    plt.hlines(y=0, xmin=0, xmax=20)
    plt.title('Residual Plot using training (blue) and test (green) data')
    plt.ylabel('Residuals')
    plt.savefig("images/" + name + "-residuals" + dataset_short_name + ".png")

    print("Fit a model on X_train and calculate Mean-squared error with Y_train: ",
          np.mean((Y_train - pred_train) ** 2))
    print("Fit a model on X_train and calculate Mean-squared error with X_test, Y_test: ",
          np.mean((Y_test - pred_test) ** 2))
    error =  sqrt(mean_squared_error(dataset[dependent_variable], y_pred))
    print("Mean square error is {}".format(error))

    predictions = regressor.predict(dataset_test.drop("id", axis = 1))
    predictions = np.round(predictions,0).astype(int)
    return dataset_test["id"],predictions
