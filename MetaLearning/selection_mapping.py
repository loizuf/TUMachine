import pandas as pd
import scipy
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

param_dist_rf = {"max_depth": [1, 2, 3, None],
                 "n_estimators": sp_randint(1, 11),
                 "max_features": sp_randint(1, 11),
                 "min_samples_split": sp_randint(2, 15),
                 "min_samples_leaf": sp_randint(1, 11),
                 "bootstrap": [True, False],
                 "criterion": ["gini", "entropy"]}
param_dist_svm = {'C': scipy.stats.expon(scale=100),
                  'gamma': scipy.stats.expon(scale=.1),
                  'kernel': ['rbf', 'poly'],
                  'class_weight': ['balanced', None]}
param_dist_mlp = {"activation":["logistic","relu", "tanh"],
                  "alpha": scipy.stats.expon(scale=.1),
                  'num_hidden_layers': [1,2],
                    'hidden_layer_size': [10,50,100],
                  }


if __name__ == '__main__':
    meta_database = pd.read_csv("datasets_test/metaframe.csv")
    svm = SVC(C=1)
    
    # n_splits cannot be greater than the number of members in each class
    y_pred = cross_val_predict(svm, meta_database.drop("classifier", axis=1), meta_database["classifier"], cv=5)
    acc_score = accuracy_score(meta_database["classifier"], y_pred)
    dist = scipy.stats.expon(scale=.1)
    print("Accuracy:", acc_score)
