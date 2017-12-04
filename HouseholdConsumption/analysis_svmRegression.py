from HouseholdConsumption.analysis_util import calc_regression
from sklearn.svm import SVR

if __name__ == "__main__":
    calc_regression(SVR(), "SVM_Regression")