import KDDcup1998.analysis_utils as utils
from sklearn.svm import SVC

if __name__ == '__main__':
    clf = SVC(C=0.5)
    y_true, y_pred = utils.do_classification(clf)
    #utils.write_subms(y_true,y_pred,"svm")
