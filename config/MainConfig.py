from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.metrics as metrics
import tensorflow.keras.losses as losses
from os.path import join
import os
import tensorflow as tf

import AI.proj_metrics as mymetrics
from constants_proj.AI_proj_params import *
from constants.AI_params import TrainingParams, ModelParams, AiModels
from img_viz.constants import PlotMode

# train_rows = 2672
# train_cols = 4008
train_rows = 528
train_cols = 800

_run_name = F"IN_NoNorm_NET_UNET3L3F_{train_rows}x{train_cols}"

_output_folder = "/data/HYCOM/DA_HYCOM_TSIS/"  # Where to save everything
# _preproc_folder = "/data/BetyFishClassification/INPUT_RAW"
_preproc_folder = "/data/BetyFishClassification/PREPROC"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Decide which GPU to use to execute the code
# tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12288)
tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)



_output_folder = "/data/BetyFishClassification/OUTPUT//"  # Where to save everything

def get_preproc_config():
    model_config = {
        PreprocParams.input_folder_raw: "/data/BetyFishClassification/INPUT_RAW",
        PreprocParams.output_folder: _preproc_folder,
        PreprocParams.imgs_output_folder: "/data/BetyFishClassification/INPUT_MERGED",
        ProjTrainingParams.rows: train_rows,
        ProjTrainingParams.cols: train_cols,
    }
    return model_config


def append_model_params(cur_config):
    model_config = {
        ModelParams.MODEL: AiModels.HALF_UNET_2D_SINGLE_STREAM_CLASSIFICATION,
        ModelParams.DROPOUT: False,
        ModelParams.BATCH_NORMALIZATION: True,
        ModelParams.INPUT_SIZE: [train_rows, train_cols, 3],  # 4741632
        ModelParams.START_NUM_FILTERS: 16,
        ModelParams.NUMBER_LEVELS: 5,
        ModelParams.FILTER_SIZE: 3,
        ModelParams.NUMBER_OF_OUTPUT_CLASSES: 4,
        ModelParams.NUMBER_DENSE_LAYERS: 2
    }
    return {**cur_config, **model_config}

def get_training_2d():
    cur_config = {
        TrainingParams.output_folder: F"{join(_output_folder,'Training')}",
        TrainingParams.validation_percentage: .1,
        TrainingParams.test_percentage: .1,
        TrainingParams.file_name: "RESULTS.csv",

        TrainingParams.evaluation_metrics: [metrics.sparse_categorical_crossentropy, metrics.sparse_categorical_accuracy],  # Metrics to show in tensor flow in the training
        TrainingParams.loss_function: metrics.sparse_categorical_crossentropy,  # Loss function to use for the learning

        TrainingParams.optimizer: Adam(lr=0.001),  # Default values lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
        TrainingParams.batch_size: 10, # READ In this case it is not a common batch size. It indicates the number of images to read from the same file
        TrainingParams.epochs: 5000,
        TrainingParams.config_name: _run_name,
        TrainingParams.data_augmentation: True,
        ProjTrainingParams.input_folder_preproc: _preproc_folder,
        ProjTrainingParams.output_folder: join(_output_folder, "images"),
        ProjTrainingParams.rows: train_rows,
        ProjTrainingParams.cols: train_cols,
        ProjTrainingParams.output_folder_summary_models:  F"{join(_output_folder,'SUMMARY')}",
    }
    return append_model_params(cur_config)


def get_prediction_params():
    weights_folder = join(_output_folder, "Training", _run_name, "models")
    cur_config = {
        TrainingParams.config_name: _run_name,
        PredictionParams.input_folder: _preproc_folder,
        PredictionParams.output_folder: F"{join(_output_folder,'Prediction')}",
        PredictionParams.output_imgs_folder: F"{join(_output_folder,'Prediction','imgs')}",
        PredictionParams.show_imgs: False,
        PredictionParams.model_weights_file: join(weights_folder, "Simple_CNNVeryLarge_Input_All_with_Obs_No_SSH_NO_LATLON_Output_ALL_80x80_UpSampling_NoLand_Mean_Var_2020_10_29_17_10-01-0.45879732.hdf5"),
        PredictionParams.metrics: mymetrics.only_ocean_mse,
        PredictionParams.model_split_file: "",
    }

    return {**append_model_params(cur_config), **get_training_2d()}

