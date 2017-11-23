import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

train_complete = pd.read_csv("C:/Users/Riffa/Dropbox/Machine_Learning/Cancer/cleaning/breast-cancer.shuf.train.complete.csv")
train_missing_caps = pd.read_csv("C:/Users/Riffa/Dropbox/Machine_Learning/Cancer/cleaning/breast-cancer.shuf.train.missing_caps.csv")

test_complete = pd.read_csv("C:/Users/Riffa/Dropbox/Machine_Learning/Cancer/cleaning/breast-cancer.shuf.test.complete.csv")
test_missing_caps = pd.read_csv("C:/Users/Riffa/Dropbox/Machine_Learning/Cancer/cleaning/breast-cancer.shuf.test.missing_caps.csv")
test_missing_quad = pd.read_csv("C:/Users/Riffa/Dropbox/Machine_Learning/Cancer/cleaning/breast-cancer.shuf.test.missing_quad.csv")


#train_complete.drop(labels = "ID")
train_complete["age"] = train_complete["age"].astype('category')
train_complete["menopause"] = train_complete["menopause"].astype('category')
train_complete["tumor-size"] = train_complete["tumor-size"].astype('category')
train_complete["inv-nodes"] = train_complete["inv-nodes"].astype('category')
train_complete["node-caps"] = train_complete["node-caps"].astype('category')
train_complete["breast"] = train_complete["breast"].astype('category')
train_complete["breast-quad"] = train_complete["breast-quad"].astype('category')
train_complete["irradiat"] = train_complete["irradiat"].astype('category')
train_complete["Class"] = train_complete["Class"].astype('category')

cat_columns = train_complete.select_dtypes(['category']).columns
train_complete[cat_columns] = train_complete[cat_columns].apply(lambda x: x.cat.codes)

y = train_complete["node-caps"]
X_train, X_test, y_train, y_test = train_test_split(train_complete, y, test_size=0.2)

lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)


plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()

print ("Score:", model.score(X_test, y_test))

print(predictions)