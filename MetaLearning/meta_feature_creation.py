import pandas as pd
import os

from MetaLearning import model_based_characterization as mbc
from MetaLearning import preprocessing
from MetaLearning import statistical_characterization as sc

rootdir = '../datasets_test'
column_names = ["avg_na", "class_num", "attr_num", "data_num", "class_data_ratio", "class_entropy", "avg_entropy", "avg_pearsons_r", "avg_snr", "avg_sd"]

sets = preprocessing.load_sets(rootdir)
meta_frame = pd.DataFrame(columns=column_names,index=range(len(sets)))

for i in range(len(sets)):

    set_params = []
    set = sets[i]

    set_params.append(sc.nan_average_number(set))
    set_params.append(sc.unique_class_number(set))
    set_params.append(sc.attribute_number(set))
    set_params.append(sc.data_point_number(set))
    set_params.append(sc.ratio_points_to_attributes(set))

    set = preprocessing.convert_objects(set)
    set = preprocessing.fill_na(set)

    set_params.append(sc.class_entropy(set))
    set_params.append(sc.average_entropy(set))
    set_params.append(sc.average_pearsons_r(set))
    set_params.append(sc.average_signal_to_noise(set))

    set = preprocessing.min_max_scaling(set)

    set_params.append(sc.average_std_deviation(set))

    meta_frame.iloc[i] = set_params

print(meta_frame)


