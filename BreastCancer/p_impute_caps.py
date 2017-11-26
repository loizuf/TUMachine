import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

train_complete_with_ID = pd.read_csv("../datasets/Cancer/cleaning/breast-cancer.shuf.train.complete.csv")
train_missing_caps_with_ID = pd.read_csv("../datasets/Cancer/cleaning/breast-cancer.shuf.train.missing_caps.csv")

test_complete_with_ID = pd.read_csv("../datasets/Cancer/cleaning/breast-cancer.shuf.test.complete.csv")
test_missing_caps_with_ID = pd.read_csv("../datasets/Cancer/cleaning/breast-cancer.shuf.test.missing_caps.csv")


# drop ID column and convert everything to categorical data, then map numerical data on it
ages = {'20-29': 0, '30-39': 1, '40-49': 2, '50-59': 3, '60-69': 4, '70-79': 5}
meno = {'premeno': 0, 'lt40': 1, 'ge40': 2}
size = {'0-4': 0, '5-9': 1, '10-14': 2, '15-19': 3, '20-24': 4, '25-29': 5, '30-34': 6, '35-39': 7, '40-44': 8, '45-49': 9, '50-54': 10}
inv_nodes = {'0-2': 0, '3-5': 1, '6-8': 2, '9-11': 3, '12-14': 4, '15-17': 5, '24-26':6}
caps = {'no': 0, 'yes': 1}
breast = {'left': 0, 'right': 1}
quad = {'central': 0, 'left_low': 1, 'right_low': 2, 'left_up': 3, 'right_up': 4}
rad = {'no': 0, 'yes': 1}
classes = {'no-recurrence-events': 0, "recurrence-events": 1}

## Data Preparation TRAIN START

train_complete = train_complete_with_ID.drop(["ID", "Class"], axis=1)
train_complete["age"] = train_complete["age"].astype('category').map(ages)
train_complete["menopause"] = train_complete["menopause"].astype('category').map(meno)
train_complete["tumor-size"] = train_complete["tumor-size"].astype('category').map(size)
train_complete["inv-nodes"] = train_complete["inv-nodes"].astype('category').map(inv_nodes)
train_complete["node-caps"] = train_complete["node-caps"].astype('category').map(caps)
train_complete["breast"] = train_complete["breast"].astype('category').map(breast)
train_complete["breast-quad"] = train_complete["breast-quad"].astype('category').map(quad)
train_complete["irradiat"] = train_complete["irradiat"].astype('category').map(rad)

train_missing_caps = train_missing_caps_with_ID.drop(["ID", "Class", "node-caps"], axis=1)
train_missing_caps["age"] = train_missing_caps["age"].astype('category').map(ages)
train_missing_caps["menopause"] = train_missing_caps["menopause"].astype('category').map(meno)
train_missing_caps["tumor-size"] = train_missing_caps["tumor-size"].astype('category').map(size)
train_missing_caps["inv-nodes"] = train_missing_caps["inv-nodes"].astype('category').map(inv_nodes)
train_missing_caps["breast"] = train_missing_caps["breast"].astype('category').map(breast)
train_missing_caps["breast-quad"] = train_missing_caps["breast-quad"].astype('category').map(quad)
train_missing_caps["irradiat"] = train_missing_caps["irradiat"].astype('category').map(rad)

## DataPreparation TEST Start

test_complete = test_complete_with_ID.drop(["ID", "Class"], axis=1)
test_complete["age"] = test_complete["age"].astype('category').map(ages)
test_complete["menopause"] = test_complete["menopause"].astype('category').map(meno)
test_complete["tumor-size"] = test_complete["tumor-size"].astype('category').map(size)
test_complete["inv-nodes"] = test_complete["inv-nodes"].astype('category').map(inv_nodes)
test_complete["node-caps"] = test_complete["node-caps"].astype('category').map(caps)
test_complete["breast"] = test_complete["breast"].astype('category').map(breast)
test_complete["breast-quad"] = test_complete["breast-quad"].astype('category').map(quad)
test_complete["irradiat"] = test_complete["irradiat"].astype('category').map(rad)

test_missing_caps = test_missing_caps_with_ID.drop(["ID", "Class", "node-caps"], axis=1)
test_missing_caps["age"] = test_missing_caps["age"].astype('category').map(ages)
test_missing_caps["menopause"] = test_missing_caps["menopause"].astype('category').map(meno)
test_missing_caps["tumor-size"] = test_missing_caps["tumor-size"].astype('category').map(size)
test_missing_caps["inv-nodes"] = test_missing_caps["inv-nodes"].astype('category').map(inv_nodes)
test_missing_caps["breast"] = test_missing_caps["breast"].astype('category').map(breast)
test_missing_caps["breast-quad"] = test_missing_caps["breast-quad"].astype('category').map(quad)
test_missing_caps["irradiat"] = test_missing_caps["irradiat"].astype('category').map(rad)

# create logistic regression model with 80/20 split
y = train_complete["node-caps"]
templ = train_complete.drop("node-caps", axis=1)
X_train, X_test, y_train, y_test = train_test_split(templ, y, test_size=0.2)
lm = linear_model.LogisticRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

print ("Score:", model.score(X_test, y_test))
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()

# create model on complete data and use 6-fold cross-validation to test the model
lm = linear_model.LogisticRegression()
model = lm.fit(templ, y)
kf = KFold(n_splits = 5, shuffle = True)
scores = cross_val_score(model, templ, y, cv=6)
print("cross-scores: ", scores)
predictions = cross_val_predict(model, templ, y, cv=6)
plt.scatter(y, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()

# now take the complete model and predict the missing values in training and testset
print(lm.predict(train_missing_caps))
print(lm.predict(test_missing_caps))
