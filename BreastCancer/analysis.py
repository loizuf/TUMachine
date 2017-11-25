from sklearn import tree, preprocessing, metrics
import pandas as pd
import graphviz
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import timeit


#test = pd.read_csv("C:/Users/Soeren/Dropbox/Machine_Learning/Cancer/imputed/breast-cancer.shuf.test.imput.numerical.csv")
train = pd.read_csv("C:/Users/Riffa/Dropbox/Machine_Learning/Cancer/imputed/breast-cancer.shuf.train.imput.csv")
templ = pd.get_dummies(train).drop("Class_no-recurrence-events", axis=1)

#Training starts
start = timeit.default_timer()

clf = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=10)
clf = clf.fit(templ.drop("Class_recurrence-events", axis=1), templ["Class_recurrence-events"])

#Training ends
end = timeit.default_timer()
time_for_training = end-start



X_train, X_test, y_train, y_test = train_test_split(templ.drop("Class_recurrence-events", axis=1), templ["Class_recurrence-events"], test_size=0.3, shuffle=True)

predictions = clf.predict(X_test)
conf_mat = confusion_matrix(y_test, predictions)
acc_score = accuracy_score(y_test, predictions)
auc = metrics.auc(y_test, predictions, reorder=False)
print(conf_mat)
print(acc_score)
print(auc)
input()

kf = KFold(n_splits = 5, shuffle = True)

scores = cross_val_score(clf, templ.drop("Class_recurrence-events", axis=1), templ["Class_recurrence-events"], cv=10, scoring='accuracy')
print(scores)

dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=templ.drop("Class_recurrence-events", axis=1).columns.values,
                         class_names=["recurrence", "no_recurrence"],
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("tree", view=True)

print(clf)