import pandas as pd
import os

from MetaLearning import model_based_characterization as mbc
from MetaLearning import preprocessing
from MetaLearning import statistical_characterization as sc

rootdir = '../datasets_test'

sets = preprocessing.load_sets(rootdir)
for set in sets:
    set = preprocessing.convert_objects(set)
    set = preprocessing.fill_na(set)
    print(mbc.create_model_based_characteristics(set))


