import pandas as pd


def rename_first_column(dataset):
    new_columns = dataset.columns.values
    new_columns[0] = 'Time'
    dataset.columns = new_columns
    return dataset


def clear_dataset(dataset):
    # remove clients which aren't measured for the whole time window (i.e. started after 2011)
    return dataset.dropna(axis=1, how='any')


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
    file_name = "../datasets/electricity_load/LD2011_2014.txt"
    dataset = pd.read_csv(file_name, sep = ';', low_memory=False, na_values = 0)

    dataset = rename_first_column(dataset)

    dataset = clear_dataset(dataset)

    # take just first client - we must model them separately anyhow
    dataset = dataset[dataset.columns[0:2]]

    dataset = normalize_dataset(dataset)

    preprocessed_file_name = "../datasets/electricity_load/LD2011_2014_preprocessed.txt"
    write_to_csv(dataset, preprocessed_file_name)
