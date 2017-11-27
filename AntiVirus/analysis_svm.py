import AntiVirus.analysis_utils as utils
from sklearn.svm import SVC

if __name__ == '__main__':

    clf = SVC()
    utils.do_classification(clf)