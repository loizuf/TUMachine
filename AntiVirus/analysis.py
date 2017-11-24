import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

df = pd.read_csv("G:/OrderForLinux/Universitaet/Wien/3.Semester/Machine_Learning/Virus/Dataset/dataset_cleaned_minimal.csv")

y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

print(X_test)