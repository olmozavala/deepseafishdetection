from preproc.UtilsDates import get_days_from_month
from constants_proj.AI_proj_params import PreprocParams
import pandas as pd
from pandas import DataFrame
import numpy.ma as ma
from os.path import join, isfile
import os
import numpy as np

def get_all_files(input_folder):
    classes_folder = os.listdir(input_folder)
    classes_folder.sort()
    print(F"All classes are: {classes_folder}")
    all_files = []
    all_paths = []
    Y = []
    last_id = 0
    class_idxs = []
    for i_class, c_class_folder in enumerate(classes_folder):
        class_files = os.listdir(join(input_folder, c_class_folder))
        class_files.sort()
        # Saving the number of files per class
        class_idxs.append(last_id + len(class_files))
        last_id = class_idxs[i_class]
        all_files += class_files
        all_paths += [join(input_folder, c_class_folder, x) for x in class_files]
        Y += [i_class for x in class_files]

    return np.array(all_files), np.array(all_paths), np.array(Y), np.array(class_idxs), classes_folder
