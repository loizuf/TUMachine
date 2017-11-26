from sklearn import tree, metrics
import pandas as pd
import graphviz
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import timeit

# test = pd.read_csv("C:/Users/Soeren/Dropbox/Machine_Learning/Cancer/imputed/breast-cancer.shuf.test.imput.numerical.csv")
train = pd.read_csv("C:/Users/Soeren/Dropbox/Machine_Learning/Cancer/imputed/breast-cancer.shuf.train.imput.csv")
templ = pd.get_dummies(train).drop("Class_no-recurrence-events", axis=1)
print(templ.head)
X_train, X_test, y_train, y_test = train_test_split(templ.drop("Class_recurrence-events", axis=1),
                                                    templ["Class_recurrence-events"], test_size=0.3, shuffle=True)

# Training 70/30 split starts
start = timeit.default_timer()

clf = tree.DecisionTreeClassifier(min_impurity_decrease=0.01, min_samples_leaf=4)
clf = clf.fit(X_train, y_train)

# Training 70/30 split ends
end = timeit.default_timer()
time_for_training = end - start

# Evaluation 70/30 split starts
start = timeit.default_timer()

predictions = clf.predict(X_test)
conf_mat = confusion_matrix(y_test, predictions)
acc_score = accuracy_score(y_test, predictions)
auc = metrics.auc(y_test, predictions, reorder=False)
print(conf_mat)
print(acc_score)
print(auc)

# Evaluation 70/30 split ends
end = timeit.default_timer()
time_for_split_eval = end - start

# Evaluation 10-fold cross starts
start = timeit.default_timer()

kf = KFold(n_splits=10, shuffle=True)
fold_scores = cross_val_score(clf, templ.drop("Class_recurrence-events", axis=1), templ["Class_recurrence-events"],
                              cv=10, scoring='accuracy')
print(fold_scores)

# Evaluation 10-fold cross ends
end = timeit.default_timer()
time_for_fold_eval = end - start

# Visualization (no timing, why thank you I'd like a kangaroo)
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=templ.drop("Class_recurrence-events", axis=1).columns.values,
                                class_names=["no recurrence", "recurrence"],
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("tree",view=True)

# Write results to file
dt_results = open("analysis/dt.txt", "w")
dt_results.write("These are the Results of analyzing the Breast-Dataset with the following decision Tree:\n\n" + str(clf)
                 + "\n\nThe following measurements were taken:"
                 + "\nConfusion matrix: \n" + str(conf_mat)
                 + "\nAccuracy: " + str(acc_score)
                 + "\nArea under the Curve: " + str(auc)
                 + "\n10-fold scores:\n " + str(fold_scores)
                 + "\nTraining (70/30 split) took: " + str(time_for_training)
                 + " seconds\nEvaluation (70/30 split) took: " + str(time_for_split_eval)
                 + " seconds\nEvaluation(10-fold) took: " + str(time_for_fold_eval) + " seconds")