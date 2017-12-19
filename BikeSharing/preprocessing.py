import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import FunctionTransformer
import numpy as np


def clear_dataset(dataset):
    # remove clients which aren't measured for the whole time window (i.e. started after 2011)
    return dataset.dropna(axis=1, how='any')


def write_to_csv(dataset, csv_file):
    dataset.to_csv(csv_file, index=False, encoding='utf-8')


def log_transform_dataset(dataset):
    transformer = FunctionTransformer(np.log1p)
    return transformer.transform(dataset)


def exp_transform_dataset(dataset):
    transformer = FunctionTransformer(np.exp)
    return transformer.transform(dataset)


def center_dataset(dataset, test):
    # assumes that data has a Gaussian (bell curve) distribution
    # source: https://machinelearningmastery.com/normalize-standardize-machine-learning-data-weka/
    '''
    if(test=="test"):
        numeric_columns = dataset.select_dtypes(exclude=['object']).drop("id", axis=1).columns.values
        dataset[numeric_columns] = dataset.select_dtypes(exclude=['object']).drop("id", axis=1).apply(lambda x: (x - x.mean()))
    else:
        numeric_columns = dataset.select_dtypes(exclude=['object']).drop("cnt", axis=1).columns.values
        dataset[numeric_columns] = dataset.select_dtypes(exclude=['object']).drop("cnt", axis=1).apply(lambda x: (x - x.mean()))'''

    numeric_columns = dataset.select_dtypes(exclude=['object']).columns.values
    dataset[numeric_columns] = dataset.select_dtypes(exclude=['object']).apply(
        lambda x: (x - x.mean()))
    return dataset


def standardize_dataset(dataset, test):
    # assumes that data has a Gaussian (bell curve) distribution
    # source: https://machinelearningmastery.com/normalize-standardize-machine-learning-data-weka/
    '''
    if (test == "test"):
        numeric_columns = dataset.select_dtypes(exclude=['object']).drop("id", axis=1).columns.values
        dataset[numeric_columns] = dataset.select_dtypes(exclude=['object']).drop("id", axis=1).apply(
            lambda x: (x - x.mean())/ x.std())
    else:
        numeric_columns = dataset.select_dtypes(exclude=['object']).drop("cnt", axis=1).columns.values
        dataset[numeric_columns] = dataset.select_dtypes(exclude=['object']).drop("cnt", axis=1).apply(
            lambda x: (x - x.mean())/ x.std())'''

    numeric_columns = dataset.select_dtypes(exclude=['object']).columns.values
    dataset[numeric_columns] = dataset.select_dtypes(exclude=['object']).apply(
        lambda x: (x - x.mean()) / x.std())
    return dataset


def normalize_dataset(dataset, test):
    # scaling to [0,1] interval

    '''
    if (test == "test"):
        numeric_columns = dataset.select_dtypes(exclude=['object']).drop("id", axis=1).columns.values
    else:
        numeric_columns = dataset.select_dtypes(exclude=['object']).drop("cnt", axis=1).columns.values'''

    numeric_columns = dataset.select_dtypes(exclude=['object']).columns.values
    result = dataset.copy()
    for feature_name in dataset[numeric_columns]:
        max_value = dataset[feature_name].max()
        min_value = dataset[feature_name].min()
        result[feature_name] = (dataset[feature_name] - min_value) / (max_value - min_value)
    return result


def test_for_distribution(dataset):
    # is our data normaly distributed
    # https://stackoverflow.com/questions/12838993/scipy-normaltest-how-is-it-used
    numeric_columns = dataset.select_dtypes(exclude=['object'])


def preprocessing(test_or_train):
    file_name = "../datasets/Bikesharing/bikeSharing.shuf."+test_or_train+".csv"
    dataset = pd.read_csv(file_name, sep=',', low_memory=False)
    # dataset = clear_dataset(dataset) # this set has no NAs

    # take two clients - we must model them separately anyhow
    # dataset = dataset[dataset.columns[0:12:5]]

    # dataset = prepare_values(dataset)

    if (test_or_train == "train"):
        dataset = dataset.drop(columns=["id", "dteday"])
    else:
        dataset = dataset.drop(columns=["dteday"])

    #This is because of the assumption that the variable to predict is not normally distributed (which it isn't)
    '''
    temp = dataset['cnt']
    v = dataset.columns.values
    dataset = pd.DataFrame(exp_transform_dataset(dataset))
    dataset.columns = v
    dataset['cnt'] = temp'''

    test_for_distribution(dataset)

    dataset_norm = normalize_dataset(dataset, test_or_train)
    dataset_stand = standardize_dataset(dataset, test_or_train)
    dataset_center = center_dataset(dataset, test_or_train)

    preprocessed_file_name_norm = "../datasets/Bikesharing/bikeSharing.shuf."+test_or_train+".norm.csv"
    preprocessed_file_name_stand = "../datasets/Bikesharing/bikeSharing.shuf."+test_or_train+".stand.csv"
    preprocessed_file_name_center = "../datasets/Bikesharing/bikeSharing.shuf."+test_or_train+".center.csv"
    write_to_csv(dataset_norm, preprocessed_file_name_norm)
    write_to_csv(dataset_stand, preprocessed_file_name_stand)
    write_to_csv(dataset_center, preprocessed_file_name_center)

if __name__ == '__main__':
    preprocessing("test")
    preprocessing("train")
