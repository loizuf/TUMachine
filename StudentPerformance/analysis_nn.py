from StudentPerformance.analysis_util import calc_regression, write_to_file
from sklearn.neural_network import MLPRegressor


if __name__ == "__main__":
    method_name = "neural_network"
    ids, predictions = calc_regression(MLPRegressor((10,15,10), max_iter=200), method_name)
    write_to_file(ids,predictions, method_name)