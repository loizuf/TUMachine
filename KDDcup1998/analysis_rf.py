from sklearn.ensemble import RandomForestClassifier

import KDDcup1998.analysis_utils as utils

if __name__ == '__main__':
    clf = RandomForestClassifier(min_samples_leaf=1000, max_depth=2)
    y_true, y_pred = utils.do_classification(clf)
    utils.write_subms(y_true,y_pred,"rf")
