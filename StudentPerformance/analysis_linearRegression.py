from StudentPerformance.analysis_util import calc_regression, write_to_file
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
   ids, predictions = calc_regression(LinearRegression(), "linear_regression")
   write_to_file(ids, predictions, "lin_reg")