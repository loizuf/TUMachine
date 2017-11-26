import pandas as pd
import csv


def fix_files_csv(file1_name, file2_name):
    file1 = open(file1_name, "r")
    file2 = open(file2_name, "w")

    for row in file1:
        attributes = row.replace(" ", "").split(",")
        pamwriter = csv.writer(file2)
        pamwriter.writerow(attributes)
    file1.close()
    file2.close()


def preprocess_dataset(dataset):
    # drop ID attribute
    dataset.drop('CONTROLN', axis=1, inplace=True)

    # remove attributes with more then 40% of missing values
    # keep if you have more then 60% of values
    feasible_columns = dataset.count() > 0.60 * len(dataset.index)
    # selects columns where boolean_mask is True
    dataset = dataset.loc[:, feasible_columns]

    # remove constant attributes
    notunique_columns = dataset.apply(pd.Series.nunique) != 1
    dataset = dataset.loc[:, notunique_columns]

    # clean ZIP attribute, selected objects and looked at number of different values
    # saw 144 values with "-" in dataset
    dataset['ZIP'] = pd.to_numeric(dataset['ZIP'].apply(lambda x: x.rstrip("-")))

    # removing the data with 'sparse' distribution
    # X    2492
    # 1       4
    # 5       2
    # 2       2
    # Name: MDMAUD_F
    dataset.drop("MDMAUD_F", axis=1, inplace=True)  # sure
    dataset.drop("MDMAUD_R", axis=1, inplace=True)
    dataset.drop("MDMAUD_A", axis=1, inplace=True)
    # XXXX    2492
    # C1CM       2
    # D5CM       2
    # I1CM       1
    # D2CM       1
    # L1CM       1
    # C2CM       1
    # Name: MDMAUD
    dataset.drop("MDMAUD", axis=1, inplace=True)
    dataset.drop("NOEXCH", axis=1, inplace=True)

    return dataset


def find_attributes_and(dataset1, dataset2):
    # make join of columns from train and test databases
    attribute_dataset_test = set(dataset_test.columns.values.tolist())
    attribute_dataset_train = set(dataset_train.columns.values.tolist())
    return [x for x in attribute_dataset_test if x in attribute_dataset_train]


def use_only_and_attributes(dataset, attributes_and):
    dataset2 = pd.DataFrame(columns=[])
    for attribute in dataset:
        if attribute in attributes_and:
            dataset2[attribute] = dataset[attribute]
    return dataset2


def impute_data(dataset):
    # data imputation with mean of numeric variables and use prev and succ value
    # we chose median because, otherwise some of the in value become float
    return dataset.fillna(dataset.median()).fillna(method="pad").fillna(method="bfill")


def write_to_csv(dataset, csv_file):
    dataset.to_csv(csv_file, encoding='utf-8')


if __name__ == '__main__':
    file_name_train = "/home/felentovic/Documents/TUWien/Semester_3/Machine_Learning/Excercise1/cup98ID.shuf.5000.train.csv"
    file_name_train2 = file_name_train[:-4] + "2.csv"

    fix_files_csv(file_name_train, file_name_train2)
    dataset_train = pd.read_csv(file_name_train2, low_memory=False)
    dataset_train = preprocess_dataset(dataset_train)

    file_name_test = "/home/felentovic/Documents/TUWien/Semester_3/Machine_Learning/Excercise1/cup98ID.shuf.5000.test.csv"
    file_name_test2 = file_name_test[:-4] + "2.csv"

    fix_files_csv(file_name_test, file_name_test2)
    dataset_test = pd.read_csv(file_name_test2, low_memory=False)
    dataset_test = preprocess_dataset(dataset_test)
    ################
    attributes_and = find_attributes_and(dataset_train, dataset_test)

    target_b = dataset_train['TARGET_B']
    dataset_train = use_only_and_attributes(dataset_train, attributes_and)
    dataset_train['TARGET_B'] = target_b

    dataset_test = use_only_and_attributes(dataset_test, attributes_and)
    ######################
    # imputation
    dataset_test = impute_data(dataset_test)
    dataset_train = impute_data(dataset_train)

    write_to_csv(dataset_train, file_name_train2)
    write_to_csv(dataset_test, file_name_test2)
