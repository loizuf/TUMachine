from BikeSharing.analysis_util import calc_regression
from sklearn.neighbors import KNeighborsRegressor


if __name__ == "__main__":
    calc_regression(KNeighborsRegressor(n_neighbors=3), "knn_regression")