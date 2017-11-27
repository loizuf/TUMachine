import csv
import time

import numpy as np
import pandas as pd
from sklearn import metrics

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import  cross_val_predict

def write_subms(dataset_y, predictions, method_name):
    subm = "cup98ID.predictions_" + method_name + ".csv"
    file_subm = open(subm, "w")
    pamwriter = csv.writer(file_subm)

    pamwriter.writerow(["CONTROLN", "TARGET_B"])
    counter = 0
    for index in range(len(predictions)):
        if predictions[index] == 1:
            counter += 1

        # print([ids[index],predictions[index]])
        pamwriter.writerow([dataset_y[index], predictions[index]])
    print("predicted 1s", counter)
    file_subm.close()



def discretize(dataset_train, dataset_test):
    for att in dataset_train:
        if att in dataset_test and dataset_train[att].dtype != "object" and not("_" in att):
            train_series, bins = pd.qcut(dataset_train[att] + jitter(dataset_train[att]), 4, retbins=True, labels=False)
            bins[0] = np.NINF
            bins[4] = np.Infinity
            test_series = pd.cut(dataset_test[att] + jitter(dataset_test[att]), bins=bins, include_lowest=True, labels=False)
            dataset_train[att] = train_series
            dataset_test[att] = test_series


# https://stackoverflow.com/questions/20158597/how-to-qcut-with-non-unique-bin-edges
def jitter(a_series, noise_reduction=1000000):
    return (np.random.random(len(a_series)) * a_series.std() / noise_reduction) - (
        a_series.std() / (2 * noise_reduction))

def do_classification(clf, discretize_flag=False):
    file_name = "../datasets/KDD Cup 1998/final(numerical_normalized)/cup98ID.shuf.5000.train2.csv"
    dataset_train = pd.read_csv(file_name, low_memory=False)
    file_name = "../datasets/KDD Cup 1998/final(numerical_normalized)/cup98ID.shuf.5000.test2.csv"
    dataset_test = pd.read_csv(file_name, low_memory=False)

    if discretize_flag:
        discretize(dataset_train, dataset_test)

    y_pred = cross_val_predict(clf, dataset_train.drop("TARGET_B", axis=1), dataset_train["TARGET_B"], cv=10)

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

    conf_mat = confusion_matrix(dataset_train["TARGET_B"], y_pred)
    acc_score = accuracy_score(dataset_train["TARGET_B"], y_pred)
    print("Confusion matrix 10-fold")
    print(conf_mat)
    print("ACC_score:", acc_score)

    try:
        auc_score = metrics.auc(dataset_train["TARGET_B"], y_pred)
    except:
        auc_score = metrics.auc(dataset_train["TARGET_B"], y_pred, reorder=True)

    print("AUC: ", auc_score)
    return dataset_test["CONTROLN"], predictions
