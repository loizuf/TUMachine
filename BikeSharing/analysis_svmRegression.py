from BikeSharing.analysis_util import calc_regression
from sklearn.svm import SVR

if __name__ == "__main__":
    calc_regression(SVR(C=0.5), "SVM_Regression")