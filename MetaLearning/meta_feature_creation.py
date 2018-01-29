import pandas as pd
import sklearn as sk
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from MetaLearning import preprocessing
from MetaLearning import statistical_characterization as sc


def get_best_classifier(data_set):
    x_train, x_test, y_train, y_test = train_test_split(data_set.drop(['class'], axis=1), data_set['class'].astype('int'), test_size = 0.3, random_state = 0)
    classifiers = [
        DecisionTreeClassifier(max_depth=5, max_leaf_nodes=20),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        KNeighborsClassifier(2),
        MLPClassifier(hidden_layer_sizes=(5,2)),
        MultinomialNB()
        ]
    max_score = 0
    for i in range(len(classifiers)):
        clf = classifiers[i]
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        if score > max_score:
            max_score = score
            max_index = i
        print("finished: " + str(i))
    return max_index

rootdir = '../datasets_test/datasets'
column_names = ["contains_cat", "avg_na", "class_num", "attr_num", "data_num", "class_data_ratio", "class_entropy", "avg_entropy", "avg_pearsons_r", "avg_snr", "avg_sd", "classifier"]

sets = preprocessing.load_sets(rootdir)
meta_frame = pd.DataFrame(columns=column_names,index=range(len(sets)))

for i in range(len(sets)):
    set_params = []
    set = sets[i]

    set_params.append(sc.contains_categorical(set))
    set_params.append(sc.nan_average_number(set))
    set_params.append(sc.unique_class_number(set))
    set_params.append(sc.attribute_number(set))
    set_params.append(sc.data_point_number(set))
    set_params.append(sc.ratio_points_to_attributes(set))

    set = preprocessing.convert_objects(set)
    set = preprocessing.fill_na(set)

    set_params.append(sc.class_entropy(set))
    set_params.append(sc.average_entropy(set))
    set_params.append(sc.average_pearsons_r(set))
    set_params.append(sc.average_signal_to_noise(set))

    set = preprocessing.min_max_scaling(set)

    set_params.append(sc.average_std_deviation(set))
    set_params.append(get_best_classifier(set))

    meta_frame.iloc[i] = set_params
    print("done with set: " + str(i))

meta_frame.to_csv("../datasets_test/metaframe.csv")



