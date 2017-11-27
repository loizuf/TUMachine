import CarEvaluation.analysis_utils as utils
from sklearn.naive_bayes import MultinomialNB

if __name__ == '__main__':

    clf = MultinomialNB()
    utils.do_classification(clf)