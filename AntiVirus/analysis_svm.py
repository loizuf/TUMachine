import AntiVirus.analysis_utils as utils
from sklearn.svm import SVC

if __name__ == '__main__':

    clf = SVC(C=2)
    utils.do_classification(clf)