import time

import pandas as pd
from sklearn.naive_bayes import GaussianNB

import KDDcup1998.utils_cup as utils_cup
import utils.utils as utils
import numpy as np

def discretize(dataset_train, dataset_test):
    for att in dataset_train:
        if att in dataset_test and dataset_train[att].dtype != "object":
            train_series, bins = pd.qcut(dataset_train[att] + jitter(dataset_train[att]), 4, retbins=True)
            test_series = pd.cut(dataset_test[att] + jitter(dataset_test[att]), bins=bins, include_lowest=True)
            dataset_train[att] = train_series
            dataset_test[att] = test_series

#https://stackoverflow.com/questions/20158597/how-to-qcut-with-non-unique-bin-edges
def jitter(a_series, noise_reduction=1000000):

    return (np.random.random(len(a_series))*a_series.std()/noise_reduction)-(a_series.std()/(2*noise_reduction))


if __name__ == '__main__':
    file_name = "/home/felentovic/Documents/TUWien/Semester_3/Machine_Learning/Excercise1/cup98ID.shuf.5000.train2.csv"
    dataset_train = pd.read_csv(file_name, low_memory=False)
    clf = GaussianNB()
    file_name = "/home/felentovic/Documents/TUWien/Semester_3/Machine_Learning/Excercise1/cup98ID.shuf.5000.test2.csv"
    dataset_test = pd.read_csv(file_name, low_memory=False)

    discretize(dataset_train, dataset_test)


    utils.prepare_one_hot_enc(dataset_train, dataset_test)
    dataset_test = pd.get_dummies(dataset_test)
    dataset_train = pd.get_dummies(dataset_train)

    # kf = KFold(n_splits=10, shuffle=True)
    # fold_scores = cross_val_score(gnb, dataset_train.drop("TARGET_B", axis=1), dataset_train["TARGET_B"],
    #                               cv=10, scoring='f1_micro')
    # print(fold_scores)

    print("FITTING")
    start = time.time()
    clf.fit(dataset_train.drop("TARGET_B", axis=1), dataset_train["TARGET_B"])
    end = time.time()
    print("finished in", end - start)

    print("PREDICTING")
    start = time.time()
    predictions = clf.predict(dataset_test.drop("CONTROLN", axis=1))
    end = time.time()
    print("finished in", end - start)

    utils_cup.write_subms(dataset_test, predictions, "nb")
