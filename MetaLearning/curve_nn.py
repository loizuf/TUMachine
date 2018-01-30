import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import validation_curve
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':
    meta_database = pd.read_csv("datasets_test/metaframe.csv")
    classifiers = meta_database["classifier"]
    meta_database = meta_database.drop("id", axis=1)
    meta_database = meta_database.drop("classifier", axis=1)
    # normalize
    normalized_df = (meta_database - meta_database.mean()) / meta_database.std()
    X, y = normalized_df, classifiers

    nn_params = {'activation': 'logistic', 'hidden_layer_sizes': (10, 5), 'learning_rate_init': 0.010704907777402024,
                 'alpha': 0.00019455756835096155}
    nn = MLPClassifier(**nn_params)
    alpha_range = np.logspace(-5, -2, 5)

    train_scores, test_scores = validation_curve(
        nn, X, y, param_name="alpha", param_range=alpha_range,
        cv=5, scoring="accuracy", n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Accuracy Curve with Neural Network")
    plt.xlabel("alpha")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(alpha_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(alpha_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(alpha_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(alpha_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
