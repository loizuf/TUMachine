from sklearn import tree, preprocessing
import pandas as pd
import graphviz

test = pd.read_csv("C:/Users/Soeren/Dropbox/Machine_Learning/Cancer/imputed/breast-cancer.shuf.test.imput.numerical.csv")
train = pd.read_csv("C:/Users/Soeren/Dropbox/Machine_Learning/Cancer/imputed/breast-cancer.shuf.train.imput.csv")


templ = pd.get_dummies(train).drop("Class_no-recurrence-events", axis=1)

print(list(templ.columns.values))
input()

'''
enc = preprocessing.OneHotEncoder()
enc.fit(train)
onehotlabels = enc.transform(train).toarray()
print(onehotlabels.shape)
input()
'''

clf = tree.DecisionTreeClassifier()
clf = clf.fit(templ.drop("Class_recurrence-events", axis=1), templ["Class_recurrence-events"])

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("tree", view=True)

print(clf)