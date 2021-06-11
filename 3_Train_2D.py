from datetime import datetime

from config.MainConfig import get_training_2d
from AI.data_generation.Generator import data_gen_from_preproc

from constants_proj.AI_proj_params import ProjTrainingParams
import trainingutils as utilsNN
from models.modelSelector import select_2d_model
from models_proj.models import *
from img_viz.common import create_folder
from io_project.read_utils import get_all_files

from os.path import join
import numpy as np
import os
from constants.AI_params import TrainingParams, ModelParams

from tensorflow.keras.utils import plot_model

def doTraining(conf):
    input_folder_preproc = config[ProjTrainingParams.input_folder_preproc]
    # output_field = config[ProjTrainingParams.output_fields]

    output_folder = config[TrainingParams.output_folder]
    val_perc = config[TrainingParams.validation_percentage]
    test_perc = config[TrainingParams.test_percentage]
    eval_metrics = config[TrainingParams.evaluation_metrics]
    loss_func = config[TrainingParams.loss_function]
    batch_size = config[TrainingParams.batch_size]
    epochs = config[TrainingParams.epochs]
    run_name = config[TrainingParams.config_name]
    optimizer = config[TrainingParams.optimizer]

    output_folder = join(output_folder, run_name)
    split_info_folder = join(output_folder, 'Splits')
    parameters_folder = join(output_folder, 'Parameters')
    weights_folder = join(output_folder, 'models')
    logs_folder = join(output_folder, 'logs')
    create_folder(split_info_folder)
    create_folder(parameters_folder)
    create_folder(weights_folder)
    create_folder(logs_folder)

    # Compute how many cases
    all_files, all_paths, Ys, class_ids, c_names = get_all_files(input_folder_preproc)
    tot_examples = len(all_files)

    # ================ Split definition =================
    [train_ids, val_ids, test_ids] = utilsNN.split_train_validation_and_test(tot_examples,
                                                                             val_percentage=val_perc,
                                                                             test_percentage=test_perc,
                                                                             shuffle_ids=True)

    print(F"Train examples (total:{len(train_ids)}) :{all_files[train_ids]}")
    print(F"Validation examples (total:{len(val_ids)}) :{all_files[val_ids]}:")
    print(F"Test examples (total:{len(test_ids)}) :{all_files[test_ids]}")

    print("Selecting and generating the model....")
    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
    model_name = F'{run_name}_{now}'

    # ******************* Selecting the model **********************
    model = select_2d_model(config, last_activation=None)
    # https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50
    # model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet',
    #                                        input_shape=config[ModelParams.INPUT_SIZE],
    #                                        pooling=max, classes=4)

    plot_model(model, to_file=join(output_folder,F'{model_name}.png'), show_shapes=True)

    print("Saving split information...")
    file_name_splits = join(split_info_folder, F'{model_name}.txt')
    utilsNN.save_splits(file_name=file_name_splits, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)

    print("Compiling model ...")
    model.compile(loss=loss_func, optimizer=optimizer, metrics=eval_metrics)

    print("Getting callbacks ...")

    [logger, save_callback, stop_callback] = utilsNN.get_all_callbacks(model_name=model_name,
                                                                       early_stopping_func=F'val_{eval_metrics[0].__name__}',
                                                                       weights_folder=weights_folder,
                                                                       logs_folder=logs_folder)

    print("Training ...")
    # ----------- Using preprocessed data -------------------
    generator_train = data_gen_from_preproc(all_paths, Ys, config, train_ids)
    generator_val = data_gen_from_preproc(all_paths, Ys, config, val_ids)

    # Decide which generator to use
    model.fit_generator(generator_train, steps_per_epoch=int(np.ceil(len(train_ids)/batch_size)),
                        validation_data=generator_val,
                        validation_steps=int(np.ceil(len(val_ids)/batch_size)),
                        use_multiprocessing=False,
                        workers=1,
                        # validation_freq=10, # How often to compute the validation loss
                        epochs=epochs, callbacks=[logger, save_callback, stop_callback])


if __name__ == '__main__':
    config = get_training_2d()
    # Single training
    doTraining(config)
