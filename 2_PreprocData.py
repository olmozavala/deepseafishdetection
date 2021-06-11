from multiprocessing import Pool
from constants_proj.AI_proj_params import PreprocParams, ProjTrainingParams
import matplotlib.pyplot as plt
from config.MainConfig import get_preproc_config
from inout_common.io_common import create_folder
from os.path import join
import skimage.io as io
from PIL import Image
import os

NUM_PROC = 10

def main():
    # ----------- Parallel -------
    p = Pool(NUM_PROC)
    p.map(img_generation_all, range(NUM_PROC))

def img_generation_all(proc_id):
    """
    Makes images of the available data (Free run, DA and Observations)
    :param proc_id:
    :return:
    """
    config = get_preproc_config()
    input_folder = config[PreprocParams.input_folder_raw]
    output_folder = config[PreprocParams.output_folder]

    rows = config[ProjTrainingParams.rows]
    cols = config[ProjTrainingParams.cols]

    all_classes = os.listdir(input_folder)
    for c_class in all_classes:
        class_output_folder = join(output_folder, c_class)
        create_folder(class_output_folder)
        all_files_per_class = os.listdir(join(input_folder, c_class))
        for i_class, c_file_name in enumerate(all_files_per_class):
            if i_class % NUM_PROC == proc_id:
                c_path = join(input_folder, c_class, c_file_name)
                output_path = join(class_output_folder, F"{i_class}.png")
                im = Image.open(c_path)
                im_res = im.resize((cols,rows), Image.BILINEAR) # NEAREST, BILINEAR, BICUBIC
                im_res.save(output_path)
                # plt.imshow(im)
                # plt.title(c_class)
                # plt.show()

if __name__ == '__main__':
    main()
