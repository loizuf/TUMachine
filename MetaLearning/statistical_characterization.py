from random import random

import scipy.stats
import numpy as np

#7
def average_entropy(data_set):
    entropy = 0
    for attribute in list(data_set.drop(['class'], axis=1)):
        entropy += scipy.stats.entropy(data_set[attribute].value_counts(normalize=True), base=2)
    return entropy / (attribute_number(data_set) - 1)

#6
def class_entropy(data_set):
    return scipy.stats.entropy(data_set['class'].value_counts(normalize=True), base=2)


def nan_number_per_column(data_set):
    return data_set.isnull().sum()

#1
def nan_average_number(data_set):
    nans = nan_number_per_column(data_set)
    return (nans.sum(axis=0)/len(nans))/attribute_number(data_set)

#2
def unique_class_number(data_set):
    return len(data_set['class'].unique())

#3
def attribute_number(data_set):
    return data_set.shape[1]

#4
def data_point_number(data_set):
    return data_set.shape[0]

#5
def ratio_points_to_attributes(data_set):
    return data_point_number(data_set) / attribute_number(data_set)

def categorical_columns_number(data_set):
    pass
    #return data_set.select_dtypes(include=["number"])

#8
def average_pearsons_r(data_set):
    coefficient = 0;
    for a in data_set:
        for b in data_set:
            if a == b:
                continue;
            coefficient += scipy.stats.pearsonr(data_set[a], data_set[b])[0]
    return coefficient/attribute_number(data_set)

#9
def average_signal_to_noise(data_set):
    means = np.array(data_set.mean(axis=0), dtype=np.float)
    sds = np.array(data_set.std(axis=0), dtype=np.float)
    snr = means/sds
    return sum(snr)/len(snr)

#10
def average_std_deviation(data_set):
    sds = np.array(data_set.std(axis=0), dtype=np.float)
    return sum(sds)/len(sds)

#11
def eleventh_feature():
    return random.uniform(0, 1)
