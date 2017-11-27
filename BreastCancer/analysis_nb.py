from sklearn.naive_bayes import MultinomialNB
import BreastCancer.analysis_utils as utils

if __name__ == '__main__':
    clf = MultinomialNB()
    y_true, y_pred = utils.do_classification(clf)
    utils.write_results(y_true, y_pred,"nb")



