import scipy.stats


def average_entropy(data_set):
    entropy = 0
    for attribute in list(data_set.drop(['class'], axis=1)):
        entropy += scipy.stats.entropy(data_set[attribute].value_counts(normalize=True), base=2)
    return entropy / (attribute_number(data_set) - 1)


def class_entropy(data_set):
    return scipy.stats.entropy(data_set['class'].value_counts(normalize=True), base=2)


def nan_number_per_column(data_set):
    return data_set.isnull().sum()


def nan_average_number(data_set):
    nans = nan_number_per_column(data_set)
    return nans.sum(axis=0)/len(nans)


def unique_class_number(data_set):
    return len(data_set['class'].unique)


def attribute_number(data_set):
    return data_set.shape[1]


def data_point_number(data_set):
    return data_set.shape[0]


def ratio_points_to_attributes(data_set):
    return data_point_number(data_set) / attribute_number(data_set)
