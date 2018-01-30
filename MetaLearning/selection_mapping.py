import pandas as pd
import scipy
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import scipy.stats as stats
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import itertools

def get_all_combinations(input_list):
    powerset = []
    for L in range(0, len(input_list)+1):
      for subset in itertools.combinations(input_list, L):
        powerset.append(tuple(subset))
    return powerset

param_dist_rf = {"max_depth": [1, 2, 3, None],
                 "n_estimators": sp_randint(1, 11),
                 "max_features": sp_randint(1, 11),
                 "min_samples_split": sp_randint(2, 15),
                 "min_samples_leaf": sp_randint(1, 11),
                 "bootstrap": [True, False],
                 "criterion": ["gini", "entropy"]}

param_dist_svm = {'C': stats.expon(scale=50),
                  'gamma': stats.expon(scale=.01),
                  'kernel': ['rbf', 'poly'],
                  'class_weight': ['balanced', None]}

param_dist_mlp = {"activation":["logistic","relu", "tanh"],
                  "alpha": stats.expon(scale=.001),
                  'learning_rate_init': stats.uniform(0.001, 0.05),
                   "hidden_layer_sizes" : [(10),(20), (10,20), (10,10), (5,5), (5), (10,5)] + get_all_combinations([5,10,15])

                  }




def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

if __name__ == '__main__':
    meta_database = pd.read_csv("datasets_test/metaframe.csv")
    classifiers = meta_database["classifier"]
    meta_database = meta_database.drop("id", axis=1)
    meta_database = meta_database.drop("classifier", axis=1)
    #normalize
    normalized_df = (meta_database - meta_database.mean()) / meta_database.std()

    svm = SVC()
    mlp = MLPClassifier()
    rf = RandomForestClassifier()

    # n_splits cannot be greater than the number of members in each class
    # y_pred = cross_val_predict(svm, meta_database.drop("classifier", axis=1), meta_database["classifier"], cv=5)
    # acc_score = accuracy_score(meta_database["classifier"], y_pred)
    # dist = scipy.stats.expon(scale=.1)
    # print("Accuracy:", acc_score)

    attributes = normalized_df
    class_var = classifiers

    n_iter_search = 20
    random_search_rf = RandomizedSearchCV(rf, n_jobs=-1, param_distributions=param_dist_rf,cv=5, n_iter=n_iter_search, scoring="accuracy")
    random_search_rf.fit(attributes,class_var)
    report(random_search_rf.cv_results_)


    random_search_svm = RandomizedSearchCV(svm,n_jobs=-1, param_distributions=param_dist_svm, cv=5, n_iter=n_iter_search,scoring="accuracy")
    random_search_svm.fit(attributes,class_var)
    report(random_search_svm.cv_results_)

    random_search_mlp = RandomizedSearchCV(mlp,n_jobs=-1, param_distributions=param_dist_mlp, cv=5, n_iter=n_iter_search,scoring="accuracy")
    random_search_mlp.fit(attributes,class_var)
    report(random_search_mlp.cv_results_)

