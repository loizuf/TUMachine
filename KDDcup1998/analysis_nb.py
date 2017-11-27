import KDDcup1998.analysis_utils as utils
from sklearn.naive_bayes import GaussianNB, MultinomialNB

if __name__ == '__main__':
    clf = GaussianNB()
    y_true, y_pred = utils.do_classification(clf)
    utils.write_subms(y_pred,y_true,"nb")
