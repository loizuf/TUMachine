import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, auc
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

def do_classification(clf):
    file_name = "../datasets/Car evaluation/car.data2.csv"
    dataset = pd.read_csv(file_name, low_memory=False)

    X_train, X_test, y_train, y_test = train_test_split(dataset.drop("class", axis=1),
                                                        dataset["class"], test_size=0.3, shuffle=True)

    y_pred = cross_val_predict(clf, dataset.drop("class", axis=1), dataset["class"], cv=10)

    print("FITTING")
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print("finished in", end - start)

    print("PREDICTING")
    start = time.time()
    predictions = clf.predict(X_test)
    end = time.time()
    print("finished in", end - start)

    conf_mat = confusion_matrix(dataset["class"], y_pred)
    acc_score = accuracy_score(dataset["class"], y_pred)
    try:
        auc_score = auc(y_test, predictions)
    except:
        auc_score = auc(y_test, predictions, reorder=True)
    print(conf_mat)
    print("ACC: ", acc_score)
    print("AUC:", auc_score)
    # print(auc)

