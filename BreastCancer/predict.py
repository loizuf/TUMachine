import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import timeit

test = pd.read_csv("../datasets/Cancer/imputed/breast-cancer.shuf.test.imput.csv")
train = pd.read_csv("../datasets/Cancer/imputed/breast-cancer.shuf.train.imput.csv")

templ_train = pd.get_dummies(train).drop(["ID", "Class_no-recurrence-events"], axis=1)
templ_test = pd.get_dummies(test).drop("ID", axis=1)

# Training starts
start = timeit.default_timer()

clf = RandomForestClassifier(max_depth=3, min_samples_leaf=3)
clf = clf.fit(templ_train.drop("Class_recurrence-events", axis=1), templ_train["Class_recurrence-events"])

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
sub.write("\"ID\",\"Class\"\n")
for i in range(len(predictions)):
    if (predictions[i] == 0):
        sub.write(str(test["ID"][i]) + ",\"no-recurrence-event\"\n")
    else:
        sub.write(str(test["ID"][i]) + ",\"recurrence-event\"\n")
sub.close()
