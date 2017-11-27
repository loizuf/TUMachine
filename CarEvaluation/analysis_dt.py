from sklearn import tree, metrics
import pandas as pd
import graphviz
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import timeit

from sklearn.model_selection import cross_val_predict

train = pd.read_csv("C:/Users/Soeren/Dropbox/Machine_Learning/Car evaluation/car.data2.csv")

clf = tree.DecisionTreeClassifier(min_impurity_decrease=0.0005, min_samples_leaf=10)

# Evaluation 10-fold cross starts
start = timeit.default_timer()

kf = KFold(n_splits=10, shuffle=True)
y_pred = cross_val_predict(clf, train.drop("class", axis=1), train["class"],cv=10)

fold_scores = accuracy_score(train["class"], y_pred)
#auc = metrics.auc(train["class"], y_pred, reorder=True)
conf_mat = confusion_matrix(train["class"], y_pred)

# Evaluation 10-fold cross ends
end = timeit.default_timer()
time_for_fold_eval = end - start
'''
# Visualization (no timing, why thank you I'd like a kangaroo)
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=templ.drop("class", axis=1).columns.values,
                                class_names=["acc", "good", "unacc", "vgood"],
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("images/tree", view=True)'''

# Write results to file
dt_results = open("analysis/dt.txt", "w")
dt_results.write("These are the Results of analyzing the Car-evaluation-Dataset with the following decision Tree:\n\n" + str(clf)
                 + "\n\nThe following measurements were taken:"
                 + "\nConfusion matrix (added 10-fold): \n" + str(conf_mat)
                 #+ "\nAre under the Curve: " + str(auc)
                 + "\n10-fold score: " + str(fold_scores)
                 + "\nProcess took: " + str(time_for_fold_eval) + " seconds")
