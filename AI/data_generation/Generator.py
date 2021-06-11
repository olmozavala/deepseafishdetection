import numpy as np
from os.path import join, exists
import os
from constants_proj.AI_proj_params import ProjTrainingParams
from constants.AI_params import TrainingParams
from constants.AI_params import AiModels, ModelParams
from img_viz.common import create_folder

import skimage.io as io

def data_gen_from_preproc(all_files, Ys, config, ids):
    """
    This generator should generate X and Y for a CNN
    :param path:
    :param file_names:
    :return:
    """
    ex_id = -1
    np.random.shuffle(ids)
    batch_size = config[TrainingParams.batch_size]
    tot_ids = len(ids)
    da = config[TrainingParams.data_augmentation]

    # Just for debugging
    # from PIL import Image
    # output_path = join(os.path.dirname(config[TrainingParams.output_folder]),"gen_imgs")
    # create_folder(output_path)

    while True:
        try:
            succ_attempts = 0
            X = []
            Y = []
            while succ_attempts < batch_size:
                if ex_id < (tot_ids - 1): # We are not supporting batch processing right now
                    ex_id += 1
                else:
                    ex_id = 0
                    np.random.shuffle(ids) # We shuffle the folders every time we have tested all the examples

                c_id = ids[ex_id]
                try:
                    tx = io.imread(all_files[c_id])
                    ty = Ys[c_id]

                    # TODO add data augmentation here

                    if da:
                        total_filters = 1
                        # Making flipping
                        if np.random.random() <= (1.0 / total_filters):  # Only 1/3 should be flipped
                            tx = np.flip(tx,np.random.randint(0,2))

                    # Just for debugging
                    # im = Image.fromarray(tx)
                    # im.save(join(output_path, F"c_{ty}_{c_id}.png"))
                    X.append(tx)
                    Y.append(ty)

                except Exception as e:
                    print(F"Failed for {all_files[c_id]}: {e}")
                    continue

                succ_attempts += 1
            X = np.array(X)
            Y = np.array(Y)

            yield [X], [Y]
        except Exception as e:
            print(F"----- Not able to generate for file number (from batch):  {succ_attempts} ERROR: ", str(e))

