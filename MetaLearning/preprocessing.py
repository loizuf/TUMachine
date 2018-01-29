import sklearn as sk
import pandas as pd
import os


def set_names(data_set):
    return data_set.set_axis(data_set.columns.values[0:-1].tolist() + ['class'], axis='columns', inplace=False)


def load_sets(path):
    data_set_list = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            set = pd.read_csv(path+"/"+file, low_memory=False, sep=',', na_values=["", " ", "?"], header=None)
            data_set_list.append(set_names(set))
    return data_set_list


def convert_objects(data_set):
    for column in data_set:
        if data_set[column].dtype == 'object':
            data_set[column] = data_set[column].astype('category').cat.codes
    return data_set.astype('float32')


def fill_na(data_set):
    return data_set.fillna(0)


def min_max_scaling(data_set):
    scaler = sk.preprocessing.MinMaxScaler()
    scaler.fit(data_set)
    return set_names(pd.DataFrame(scaler.transform(data_set)))


def center(data_set):
    return data_set - data_set.mean()

