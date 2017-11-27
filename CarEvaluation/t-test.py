import math
from sklearn import tree, metrics
import pandas as pd
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import timeit

from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

train = pd.read_csv("../datasets/Car evaluation/car.data2.csv")



clfBase = SVC()
clfComparing = MultinomialNB()

y_base = cross_val_score(clfBase, train.drop("class", axis=1), train["class"],cv=10, scoring='accuracy')
y_comparing = cross_val_score(clfComparing, train.drop("class", axis=1), train["class"],cv=10, scoring='accuracy')

m1_avg = sum(y_base) / float(len(y_base))
m2_avg = sum(y_comparing) / float(len(y_comparing))

differences = []
for i in range(len(y_base)):
    differences.append(y_base[i] - y_comparing[i])

sum_elements = 0
diff_avg = sum(differences) / float(len(differences))
for i in range(len(differences)):
    sum_elements += (differences[i] - diff_avg)*(differences[i] - diff_avg)

sigma = math.sqrt((1.0 / 9) * sum_elements)

t = diff_avg * math.sqrt(10) / sigma
print(t)