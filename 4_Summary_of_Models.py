import os
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from config.MainConfig import get_training_2d
from constants.AI_params import TrainingParams, ModelParams
from constants_proj.AI_proj_params import ProjTrainingParams
from img_viz.common import create_folder

from ExtraUtils.NamesManipulation import *

# This code is used to generate a summary of the models
# that have been tested. Grouping by each modified parameter
def buildSummary(name, loss, path):
    split_path = join(os.path.dirname(os.path.dirname(path)), "Splits")
    split_file = os.listdir(split_path)[0]  # It is alsways assuemd there is only one split file per model
    model = [name, getNetworkTypeTxt(name), getBBOX(name), getInputFieldsTxt(name), loss, path, join(split_path,split_file)]
    return model

def buildDF(summary):
    df = {
        "Network": [x[1] for x in summary],
        "BBOX": [x[2] for x in summary],
        "InputType":[x[3] for x in summary],
        "Loss": [x[4] for x in summary],
        "Name": [x[0] for x in summary],
        "Path": [x[5] for x in summary],
        "SplitFile": [x[6] for x in summary],
    }
    return pd.DataFrame.from_dict(df)

# def fixNames(trained_models_folder):
#     print("================ Fixing files names ========================")
#     # from_txt = "IN_ALL"
#     # to_txt = "IN_Yes-STD"
#     #
#     # filter_file = "SimpleCNN"
#     # for root, dirs, files in os.walk(trained_models_folder):
#     #     for name in dirs:
#     #         if name.find(filter_file) != -1:
#     #             old_name = join(root, name)
#     #             # print(F"From {old_name} \nTo   {new_name} \n")
#     #             # os.rename(old_name, new_name)
#     #
#     # # Then all the files
#     # for root, dirs, files in os.walk(trained_models_folder):
#     #     for dirname in dirs:
#     #         print(dirname)
#     #     for name in files:
#     #         if name.find(from_txt) != -1:
#     #             old_name = join(root, name)
#     #             print(F"{old_name} \n {new_name} \n")
#     #             # os.rename(old_name, new_name)

if __name__ == '__main__':

    NET = "Network"
    IN = "InputType"
    LOSS = "Loss"

    # Read folders for all the experiments
    config = get_training_2d()
    trained_models_folder = config[TrainingParams.output_folder]
    output_folder = config[ProjTrainingParams.output_folder_summary_models]
    create_folder(output_folder)

    # fixNames("/data/HYCOM/DA_HYCOM_TSIS/Training")
    # exit()

    all_folders = os.listdir(trained_models_folder)
    all_folders.sort()
    print(all_folders)

    summary = []

    # Iterate over all the experiments
    for experiment in all_folders:
        all_models = os.listdir(join(trained_models_folder, experiment , "models"))
        min_loss = 100000.0
        best_model = {}
        # Iterate over the saved models for each experiment and obtain the best of them
        for model in all_models:
            loss = float((model.split("-")[-1]).replace(".hdf5",""))
            if loss < min_loss:
                min_loss = loss
                best_model = buildSummary(model, np.around(min_loss,5), join(trained_models_folder, experiment, "models", model))
        summary.append(best_model)
    summary = buildDF(summary)
    print(summary)

    summary.to_csv(join(output_folder,"summary.csv"))

    # ========= Compare Network type ======
    # data_novar = summary[summary[IN] == "No-STD"]  # All novar data
    # data_novar_srfhgt = data_novar[data_novar[OUT] == "SRFHGT"]  # All srfhgt data
    #
    # names = []
    # data = []
    # fig, ax = plt.subplots(figsize=(10,6))
    # for i, grp in enumerate(data_novar_srfhgt.groupby(NET)):
    #     names.append(grp[0])
    #     c_data = grp[1][LOSS].values
    #     data.append(c_data)
    #     plt.scatter(np.ones(len(c_data))*i, c_data, label=grp[0])
    #
    # plt.legend(loc="best")
    # # bp = plt.boxplot(data, labels=names, patch_artist=True, meanline=True, showmeans=True)
    # ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    # ax.set_xlabel("Network type")
    # ax.set_ylabel("Validation Loss (MSE)")
    # ax.set_title("Validation Loss by Network Type (SSH)")
    # plt.savefig(join(output_folder,F"By_Network_Type_Scatter.png"))
    # plt.show()

    print("Done!")
