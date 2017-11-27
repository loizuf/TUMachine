from sklearn import tree, metrics
import pandas as pd
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import timeit

from sklearn.model_selection import cross_val_predict

train = pd.read_csv("../datasets/KDD Cup 1998/final(numerical_normalized)/cup98ID.shuf.5000.train2.csv")

clf = RandomForestClassifier(min_impurity_decrease=0.0001, min_samples_leaf=25, class_weight={0:1, 1:10})

# Evaluation 10-fold cross starts
start = timeit.default_timer()

kf = KFold(n_splits=10, shuffle=True)
y_pred = cross_val_score(clf, train.drop("TARGET_B", axis=1), train["TARGET_B"], cv=10, scoring='accuracy')

print(y_pred)