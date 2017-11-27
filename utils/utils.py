from pandas.api.types import CategoricalDtype

def prepare_one_hot_enc(dataset_train, dataset_test):
    for att in dataset_train.select_dtypes(include=['object']):
        all_categories = set(dataset_train[att].unique())
        all_categories.update(set(dataset_test[att].unique()))
        dtype = CategoricalDtype(ordered=False, categories=all_categories)
        dataset_train[att] = dataset_train[att].astype(dtype)
        dataset_test[att] = dataset_test[att].astype(dtype)
