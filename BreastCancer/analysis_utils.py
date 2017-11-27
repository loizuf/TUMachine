import time

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import  cross_val_predict

def write_results(y_true, predictions,method):
    sub = open("submission_"+method+".csv", "w")
    sub.write("\"ID\",\"Class\"\n")
    for i in range(len(predictions)):
        if (predictions[i] == 0):
            sub.write(str(y_true[i]) + ",\"no-recurrence-events\"\n")
        else:
            sub.write(str(y_true[i]) + ",\"recurrence-events\"\n")
    sub.close()

def do_classification(clf):
    file_name = "../datasets/Cancer/imputed/breast-cancer.shuf.train.imput.numerical.csv"
    dataset_train = pd.read_csv(file_name, low_memory=False)

    file_name = "../datasets/Cancer/imputed/breast-cancer.shuf.test.imput.numerical.csv"
    dataset_test = pd.read_csv(file_name, low_memory=False)

    y_pred = cross_val_predict(clf, dataset_train.drop("Class", axis=1), dataset_train["Class"], cv=10)

    print("FITTING on training set")
    start = time.time()
    clf.fit(dataset_train.drop("Class", axis=1), dataset_train["Class"])
    end = time.time()
    print("finished in", end - start)

    print("PREDICTING on test set")
    start = time.time()
    predictions = clf.predict(dataset_test)
    end = time.time()
    print("finished in", end - start)

    conf_mat = confusion_matrix(dataset_train["Class"], y_pred)
    acc_score = accuracy_score(dataset_train["Class"], y_pred)
    print("Confusion matrix 10-fold")
    print(conf_mat)
    print("ACC_score",acc_score)

    return dataset_test["ID"], predictions
