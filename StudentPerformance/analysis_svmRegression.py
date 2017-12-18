from StudentPerformance.analysis_util import calc_regression, write_to_file
from sklearn.svm import SVR

if __name__ == "__main__":
    ids, predictions = calc_regression(SVR(kernel="rbf", C=3), "SVM_Regression")
    write_to_file(ids, predictions, "SVM")