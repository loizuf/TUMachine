import csv

import pandas as pd


def clear_dataset(dataset):
    # remove attributes with more then 40% of missing values
    # keep if you have more then 60% of values
    feasible_columns = dataset.count() > 0.60 * len(dataset.index)
    # selects columns where boolean_mask is True
    dataset = dataset.loc[:, feasible_columns]

    # remove constant attributes
    notunique_columns = dataset.apply(pd.Series.nunique) != 1
    dataset = dataset.loc[:, notunique_columns]

    return dataset


def impute_data(dataset):
    # data imputation with mean of numeric variables and use prev and succ value
    # we chose median because, otherwise some of the in value become float
    return dataset.fillna(dataset.median())


def write_to_csv(dataset, csv_file):
    dataset.to_csv(csv_file, index=False, encoding='utf-8')


def standardize_dataset(dataset):
    # assumes that data has a Gaussian (bell curve) distribution
    # source: https://machinelearningmastery.com/normalize-standardize-machine-learning-data-weka/
    numeric_columns = dataset.select_dtypes(exclude=['object']).columns.values
    dataset[numeric_columns] = dataset.select_dtypes(exclude=['object']).apply(lambda x: (x - x.mean()) / x.std())


def normalize_dataset(dataset):
    # scaling to [0,1] interval
    numeric_columns = dataset.select_dtypes(exclude=['object']).columns.values
    result = dataset.copy()
    for feature_name in dataset[numeric_columns]:
        max_value = dataset[feature_name].max()
        min_value = dataset[feature_name].min()
        result[feature_name] = (dataset[feature_name] - min_value) / (max_value - min_value)
    return result


if __name__ == '__main__':
    file_name = "../datasets/householdConsumption/household_power_consumption.txt"
    dataset = pd.read_csv(file_name, sep = ';', low_memory=False)

    dataset = clear_dataset(dataset)

    dataset = impute_data(dataset)

    dataset = normalize_dataset(dataset)

    preprocessed_file_name = "../datasets/householdConsumption/household_power_consumption_preprocessed.txt"
    write_to_csv(dataset, preprocessed_file_name)
