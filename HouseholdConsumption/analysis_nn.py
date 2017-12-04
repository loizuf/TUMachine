from HouseholdConsumption.analysis_util import calc_regression
from sklearn.neural_network import MLPRegressor


if __name__ == "__main__":
    calc_regression(MLPRegressor((10,10,10,10,10,10)), "neural_network")