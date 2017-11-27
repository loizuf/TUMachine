from sklearn.svm import SVC
import BreastCancer.analysis_utils as utils

if __name__ == '__main__':
    clf = SVC(C=0.5)
    y_true, y_pred = utils.do_classification(clf)
    utils.write_results(y_true, y_pred,"svm")

3


