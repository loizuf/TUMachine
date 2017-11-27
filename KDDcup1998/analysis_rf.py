from sklearn import tree, metrics
import pandas as pd
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import timeit

from sklearn.model_selection import cross_val_predict

train = pd.read_csv("C:/Users/Soeren/Dropbox/Machine_Learning/KDD Cup 1998/final(numerical_normalized)/cup98ID.shuf.5000.train2.csv")

clf = RandomForestClassifier(min_impurity_decrease=0.0001, min_samples_leaf=25, class_weight={0:1, 1:10})

# Evaluation 10-fold cross starts
start = timeit.default_timer()

kf = KFold(n_splits=10, shuffle=True)
y_pred = cross_val_predict(clf, train.drop("TARGET_B", axis=1), train["TARGET_B"], cv=10)

fold_scores = accuracy_score(train["TARGET_B"], y_pred)
auc = metrics.auc(train["TARGET_B"], y_pred, reorder=True)
conf_mat = confusion_matrix(train["TARGET_B"], y_pred)

# Evaluation 10-fold cross ends
end = timeit.default_timer()
time_for_fold_eval = end - start

'''
# Visualization (no timing, why thank you I'd like a kangaroo)
i_tree = 0
for this_tree in clf.estimators_:
    my_file = tree.export_graphviz(this_tree, out_file=None,
                                feature_names=templ.drop("TARGET_B", axis=1).columns.values,
                                class_names=True,
                                filled=True, rounded=True,
                                special_characters=True)
    graph = graphviz.Source(my_file)
    graph.render("images/tree_" + str(i_tree), view=True)
    i_tree = i_tree + 1
'''

# Write results to file
dt_results = open("analysis/rf.txt", "w")
dt_results.write("These are the Results of analyzing the KDDcup-Dataset with the following random forest:\n\n" + str(clf)
                 + "\n\nThe following measurements were taken:"
                 + "\nConfusion matrix (added 10-fold): \n" + str(conf_mat)
                 + "\nAre under the Curve: " + str(auc)
                 + "\n10-fold score: " + str(fold_scores)
                 + "\nProcess took: " + str(time_for_fold_eval) + " seconds")
