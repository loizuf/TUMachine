from HouseholdConsumption.analysis_util import calc_regression
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    calc_regression(LinearRegression(), "linear_regression")