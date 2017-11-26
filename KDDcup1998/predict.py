import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import timeit

test = pd.read_csv("C:/Users/Soeren/Dropbox/Machine_Learning/KDD Cup 1998/final (categorical + numerical)/cup98ID.shuf.5000.test2.csv")
train = pd.read_csv("C:/Users/Soeren/Dropbox/Machine_Learning/KDD Cup 1998/final (categorical + numerical)/cup98ID.shuf.5000.train2.csv")

templ_train = pd.get_dummies(train).drop(["TARGET_B", "CONTROLN"], axis=1).drop(test.columns[0], axis=1)
templ_test = pd.get_dummies(test).drop("CONTROLN", axis=1).drop(test.columns[0], axis=1)

# Training starts
start = timeit.default_timer()

clf = RandomForestClassifier(min_impurity_decrease=0.0001, min_samples_leaf=25, class_weight={0:1, 1:10})
clf = clf.fit(templ_train.drop("TARGET_B", axis=1), templ_train["TARGET_B"])

# Training ends
end = timeit.default_timer()
time_for_training = end - start

# Prediction starts
start = timeit.default_timer()

predictions = clf.predict(templ_test)

# Prediction ends
end = timeit.default_timer()
time_for_prediction = end - start

sub = open("submission.csv", "w")
sub.write("\"CONTROLN\",\"TARGET_B\"\n")
for i in range(len(predictions)):
        sub.write(str(test["CONTROLN"][i]) + ",\"" + predictions[i] + " \"\n")
sub.close()