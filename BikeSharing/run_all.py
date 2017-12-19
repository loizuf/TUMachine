from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from BikeSharing.analysis_util import calc_regression, predict_results


def test(scaling):
    calc_regression(KNeighborsRegressor(n_neighbors=5), "knn_regression_5", "norm")
    calc_regression(LinearRegression(), "linear_regression", "norm")
    calc_regression(MLPRegressor(hidden_layer_sizes=(20, 20, 20, 20, 20, 20)), "neural_network_6x20", "norm")
    calc_regression(MLPRegressor(hidden_layer_sizes=(50, 50, 50)), "neural_network_3x50", "norm")
    calc_regression(SVR(C=100), "SVM_Regression_100", "norm")


def predict():
    predict_results(KNeighborsRegressor(n_neighbors=5), "knn_regression_5", "norm")
    predict_results(LinearRegression(), "linear_regression", "norm")
    predict_results(MLPRegressor(hidden_layer_sizes=(20, 20, 20, 20, 20, 20)), "neural_network_6x20", "norm")
    predict_results(SVR(C=100), "SVM_Regression_100", "norm")


if __name__ == "__main__":
    test("stand")
    predict()

