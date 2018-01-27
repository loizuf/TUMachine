from sklearn import tree


def create_model_based_characteristics(data_set):
    clf = tree.DecisionTreeClassifier()
    clf.fit(data_set.drop(['class'], axis=1), data_set['class'])
    return {
        "depth": clf.tree_.max_depth,
        "nodes": clf.tree_.node_count,
        "avg_impurity": sum(clf.tree_.impurity)/len(clf.tree_.impurity)
    }
