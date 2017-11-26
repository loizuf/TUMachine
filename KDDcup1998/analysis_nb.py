import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score
import utils.utils as utils

if __name__ == '__main__':
    file_name = "/home/felentovic/Documents/TUWien/Semester_3/Machine_Learning/Excercise1/cup98ID.shuf.5000.train2.csv"
    dataset_train = pd.read_csv(file_name, low_memory=False)
    gnb = GaussianNB()
    file_name = "/home/felentovic/Documents/TUWien/Semester_3/Machine_Learning/Excercise1/cup98ID.shuf.5000.test2.csv"
    dataset_test = pd.read_csv(file_name, low_memory=False)

    utils.prepare_one_hot_enc(dataset_train,dataset_test)
    dataset_test = pd.get_dummies(dataset_test)
    dataset_train = pd.get_dummies(dataset_train)

    kf = KFold(n_splits=10, shuffle=True)
    fold_scores = cross_val_score(gnb, dataset_train.drop("TARGET_B", axis=1), dataset_train["TARGET_B"],
                                  cv=10, scoring='f1_micro')
    print(fold_scores)
    gnb.fit(dataset_train.drop("TARGET_B", axis=1), dataset_train["TARGET_B"])
    print(gnb.predict(dataset_test))