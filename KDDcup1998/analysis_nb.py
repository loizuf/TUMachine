import KDDcup1998.analysis_utils as utils
from sklearn.naive_bayes import GaussianNB, MultinomialNB

if __name__ == '__main__':
    clf = MultinomialNB()
    y_true, y_pred = utils.do_classification(clf, True)
    utils.write_subms(y_true,y_pred,"nb")
