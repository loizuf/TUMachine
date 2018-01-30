import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    meta_database = pd.read_csv("datasets_test/metaframe.csv")
    classifiers = meta_database["classifier"]
    meta_database = meta_database.drop("id", axis=1)
    meta_database = meta_database.drop("classifier", axis=1)
    # normalize
    normalized_df = (meta_database - meta_database.mean()) / meta_database.std()
    X , y = normalized_df, classifiers


    rf_params = {'criterion': 'gini', 'bootstrap': False, 'n_estimators': 4, 'max_features': 9, 'min_samples_leaf': 3, 'max_depth': 3}
    rf = RandomForestClassifier(**rf_params)
    min_samples_split_range = np.arange(2,15)

    train_scores, test_scores = validation_curve(
        rf, X, y, param_name="min_samples_split", param_range=min_samples_split_range,
        cv=5, scoring="accuracy", n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Accuracy Curve with RandomForrest")
    plt.xlabel("min_samples_split")
    plt.xlim(1, 15)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(min_samples_split_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(min_samples_split_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(min_samples_split_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(min_samples_split_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()