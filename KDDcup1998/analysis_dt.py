from sklearn.tree import DecisionTreeClassifier

import KDDcup1998.analysis_utils as utils

if __name__ == '__main__':
    clf = DecisionTreeClassifier(min_samples_leaf=100, max_depth=12)
    y_true, y_pred = utils.do_classification(clf)
    utils.write_subms(y_true, y_pred,"dt")
