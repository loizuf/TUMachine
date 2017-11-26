import csv
import time
import pandas as pd
from sklearn import svm

import utils.utils as utils


def write_subms(dataset_test, predictions):
    subm = "/home/felentovic/Documents/TUWien/Semester_3/Machine_Learning/Excercise1/cup98ID.predictions.csv"
    file_subm = open(subm, "w")
    pamwriter = csv.writer(file_subm)

    ids = dataset_test["CONTROLN"]
    pamwriter.writerow(["CONTROLN", "TARGET_B"])
    for index in range(len(predictions)):
        if predictions[index] == 1:
            print("1")

        # print([ids[index],predictions[index]])
        pamwriter.writerow([ids[index], predictions[index]])

    file_subm.close()


if __name__ == '__main__':
    file_name = "/home/felentovic/Documents/TUWien/Semester_3/Machine_Learning/Excercise1/cup98ID.shuf.5000.train2.csv"
    dataset_train = pd.read_csv(file_name, low_memory=False)

    file_name = "/home/felentovic/Documents/TUWien/Semester_3/Machine_Learning/Excercise1/cup98ID.shuf.5000.test2.csv"
    dataset_test = pd.read_csv(file_name, low_memory=False)

    clf = svm.SVC(class_weight={0: 1, 1: 10})
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
    print("finished in",end-start)

    print("PREDICTING")
    start = time.time()
    predictions = clf.predict(dataset_test.drop("CONTROLN", axis=1))
    end = time.time()
    print("finished in",end-start)

    write_subms(dataset_test,predictions)