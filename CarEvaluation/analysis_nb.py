import CarEvaluation.analysis_utils as utils
from sklearn.naive_bayes import MultinomialNB

if __name__ == '__main__':

    clf = MultinomialNB(alpha=0.999)
    utils.do_classification(clf)