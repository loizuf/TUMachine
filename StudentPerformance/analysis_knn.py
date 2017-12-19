from StudentPerformance.analysis_util import calc_regression, write_to_file
from sklearn.neighbors import KNeighborsRegressor


if __name__ == "__main__":
    ids, predictions =calc_regression(KNeighborsRegressor(n_neighbors=35), "knn_regression")
    write_to_file(ids, predictions, "knn")