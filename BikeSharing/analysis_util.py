import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_dataset(path, na_values = np.NaN):
    return pd.read_csv(path, low_memory=False, na_values=na_values)


def predict_results(regressor, name, stand_or_norm="stand"):
    dataset_test = load_dataset("../datasets/Bikesharing/bikeSharing.shuf.test."+stand_or_norm+".csv")
    dataset_train = load_dataset("../datasets/Bikesharing/bikeSharing.shuf.train."+stand_or_norm+".csv")

    X = dataset_train.drop('cnt', axis=1)

    regressor.fit(X, dataset_train['cnt'])

    pred_test = regressor.predict(dataset_test.drop('id', axis=1))

    pred_file = open("results/prediction_" + name + ".csv", "w")
    pred_file.write("\"id\",\"cnt\"\n")

    for i in range(len(pred_test)):
        pred_file.write(str(dataset_test["id"][i]) + "," + str(int(round(pred_test[i]))) + "\n")

    print("finished predicting with: " + name)


def calc_regression(regressor, name, scaling="stand"):
    dataset = load_dataset("../datasets/Bikesharing/bikeSharing.shuf.train." + scaling + ".csv")

    X = dataset.drop('cnt', axis = 1)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X , dataset["cnt"], test_size = 0.3, random_state = 5
    )

    # training
    regressor.fit(X_train, Y_train)
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
    plt.xlabel("Actual CNT")
    plt.ylabel("Predicted CNT")
    plt.title("Actual vs Predicted CNT Variable")
    plt.savefig("images/" + name + "_" + scaling + "-actualVsPredict.png")

    # residual plot
    plt.cla()
    plt.scatter(pred_train, pred_train - Y_train, c='b', s=1, alpha=0.5)
    plt.scatter(pred_test, pred_test - Y_test, c = 'g', s=1)
    plt.hlines(y=0, xmin=0, xmax = 1)
    plt.title('Residual Plot using training (blue) and test (green) data')
    plt.ylabel('Residuals')
    plt.savefig("images/" + name + "_" + scaling + "-residuals.png")

    file = open("results.txt", "a")
    file.write(name + ", " + scaling + "-scaling: "+ str(np.mean((Y_test-pred_test) ** 2)) + "\n")
    file.close()


if __name__ == '__main__':
    pass
