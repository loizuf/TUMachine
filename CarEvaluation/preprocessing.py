import pandas as pd
from pandas.api.types import CategoricalDtype


def to_numerical(dataset):
    dtype = CategoricalDtype(ordered=True, categories=["low", "med", "high", "vhigh"])
    dataset["buying"] = dataset["buying"].astype(dtype).cat.codes

    dtype = CategoricalDtype(ordered=True, categories=["low", "med", "high", "vhigh"])
    dataset["maint"] = dataset["maint"].astype(dtype).cat.codes

    dtype = CategoricalDtype(ordered=True, categories=["2", "3", "4", "5more"])
    dataset["doors"] = dataset["doors"].astype(dtype).cat.codes

    dtype = CategoricalDtype(ordered=True, categories=["2", "4", "more"])
    dataset["persons"] = dataset["persons"].astype(dtype).cat.codes

    dtype = CategoricalDtype(ordered=True, categories=["small", "med", "big"])
    dataset["lug_boot"] = dataset["lug_boot"].astype(dtype).cat.codes

    dtype = CategoricalDtype(ordered=True, categories=["low", "med", "high"])
    dataset["safety"] = dataset["safety"].astype(dtype).cat.codes

    dtype = CategoricalDtype(ordered=True, categories=["unacc", "acc", "vgood", "good"])
    dataset["class"] = dataset["class"].astype(dtype).cat.codes


if __name__ == '__main__':
    file_name = "/home/felentovic/Documents/TUWien/Semester_3/Machine_Learning/Excercise1/car.data.csv"
    dataset = pd.read_csv(file_name, low_memory=False)

    to_numerical(dataset)
    csv_file = file_name[:-4] + "2.csv"
    dataset.to_csv(csv_file, index=False, encoding='utf-8')
