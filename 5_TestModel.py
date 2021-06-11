import os

from tensorflow.keras.utils import plot_model
from inout.io_netcdf import read_netcdf
from os.path import join
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from img_viz.eoa_viz import EOAImageVisualizer
from config.MainConfig import get_prediction_params
from constants_proj.AI_proj_params import PredictionParams, ProjTrainingParams, PreprocParams
from models.modelSelector import select_2d_model
from models_proj.models import *
from constants.AI_params import TrainingParams, ModelParams, AiModels
from trainingutils import read_train_validation_and_test_ids
from img_viz.common import create_folder

from sklearn.metrics import mean_squared_error

from ExtraUtils.NamesManipulation import *
from ExtraUtils.VizUtilsProj import chooseCMAP
from io_project.read_utils import get_all_files
import skimage.io as io



def test_model(config):
    input_folder = config[PredictionParams.input_folder]
    output_folder = config[PredictionParams.output_folder]
    model_weights_file = config[PredictionParams.model_weights_file]
    output_imgs_folder = config[PredictionParams.output_imgs_folder]
    run_name = config[TrainingParams.config_name]

    output_imgs_folder = join(output_imgs_folder, run_name)
    create_folder(output_imgs_folder)

    # *********** Chooses the proper model ***********
    print('Reading model ....')
    model = select_2d_model(config, last_activation=None)
    plot_model(model, to_file=join(output_folder, F'running.png'), show_shapes=True)

    # *********** Reads the weights***********
    print('Reading weights ....')
    model.load_weights(model_weights_file)

    # *********** Read files to predict***********
    train_ids, val_ids, test_ids = read_train_validation_and_test_ids(config[PredictionParams.model_split_file])
    all_files, all_paths, Ys, class_ids, c_names = get_all_files(input_folder)
    [create_folder(join(output_imgs_folder, F"Class_{x}")) for x in range(len(class_ids))]

    # this array will hold all the network results
    correct_classified = [-1*np.ones(x) for x in class_ids]
    # np.random.shuffle(model_files)  # TODO this is only for testing
    np.random.shuffle(train_ids)
    for i, id_file in enumerate(np.concatenate((train_ids, val_ids, test_ids))):
        # Make the prediction of the network
        tx = io.imread(all_paths[id_file])
        ty = Ys[id_file]

        # ============================== PREDICTION ========================
        # start = time.time()
        output_nn_original = model.predict([np.array([tx])], verbose=1)
        nn_pred = np.argmax(output_nn_original)
        nn_all = [F'{x:0.2f}' for x in output_nn_original[0,:]]
        c_class = np.argmax(id_file < class_ids)
        last_value = np.argmax(correct_classified[c_class] == -1)
        correct_classified[c_class][last_value] = nn_pred == ty

        # Plot with title to veiry
        if i % 10 == 0:
            import matplotlib.pyplot as plt
            plt.imshow(tx)
            plt.title(F"Id: {id_file} True: {ty}  NN {nn_pred} \n {nn_all}")
            plt.savefig(join(output_imgs_folder, F"Class_{c_class}", F"{id_file}_{ty}"))
            plt.close()


    file_name = join(output_imgs_folder, "summary.txt")
    f = open(file_name,"w")
    error = np.array(correct_classified)
    for c_class in range(len(class_ids)):
        analyzed = error[c_class] != -1
        result = F"Corrrectly classified for class {c_names[c_class]} --> {np.sum(error[c_class][analyzed])/np.sum(analyzed):0.3f} \n"
        f.write(result)
        print(result)
    f.close()
    print("Done!")


if __name__ == '__main__':

    config = get_prediction_params()
    # -------- for all summary model testing --------------
    summary_file = "/data/BetyFishClassification/OUTPUT/SUMMARY/summary.csv"
    df = pd.read_csv(summary_file)

    for model_id in range(len(df)):
        model = df.iloc[model_id]
        # setting model weights file
        config[PredictionParams.model_weights_file] = model["Path"]
        config[PredictionParams.model_split_file] = model["SplitFile"]
        print(f"Model's weight file: {model['Path']}")
        # set the name of the network
        run_name = model['Name'].replace(".hdf5", "")
        config[TrainingParams.config_name] = run_name
        test_model(config)