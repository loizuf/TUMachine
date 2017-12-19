import pandas as pd


def categorical_to_bin(object_columns, dataset):
    for att in object_columns:
        if len(dataset[att].unique()) == 2:
            dataset[att] = dataset[att].astype('category').cat.codes


def write_to_csv(dataset, csv_file):
    dataset.to_csv(csv_file, index=False, encoding='utf-8')

def pre_norm(dataset):
    dataset["id"] = dataset["id"].astype('object')
    if 'Grade' in dataset:
        dataset['Grade'] = dataset['Grade'].astype('object')


def post_norm(dataset):
    dataset["id"] = dataset["id"].astype('int64')
    if 'Grade' in dataset:
        dataset['Grade'] = dataset['Grade'].astype('int64')


if __name__ == '__main__':

    file_name = "../datasets/Student_Performance/StudentPerformance.shuf.test.csv"
    dataset = pd.read_csv(file_name, low_memory=False)


    pre_norm(dataset)
    #normalize data
    numeric_columns = dataset.select_dtypes(exclude=["object"]).columns.values

    #dataset[numeric_columns] = dataset[numeric_columns].apply(lambda x :(x - x.min()) /(x.max() - x.min()))

    dataset[numeric_columns] = dataset[numeric_columns].apply(lambda x :(x - x.mean()) /(x.std()))

    # standardize data
    post_norm(dataset)


    ##select object columns
    object_columns = dataset.select_dtypes(include=['object']).columns.values
    categorical_to_bin(object_columns, dataset)
    ##one hot encoding
    dataset = pd.get_dummies(dataset)

    write_to_csv(dataset,file_name[:-4]+"_preprocessedSTD.csv")
