import glob
import os
import shutil
import sys
import argparse
from tqdm import tqdm


import numpy as np
from scipy import misc

def get_cam3_files(files):
    is_cam3 = lambda x: x.find('cam3_') != -1
    return sorted(list(filter(is_cam3, files)))

def get_file_id(filename):
    return filename.split('_')[1].split('.')[0]

# def delete_all_cam_files(id, path):

def copyFile(src, dest, filename):
    shutil.copyfile(os.path.join(src, filename), os.path.join(dest, filename))


def cleanupFolder(base_path):
    print("Cleaning up :: {}".format(base_path))
    
    base_orig_path = os.path.join(base_path, 'IMG')
    base_new_path = os.path.join(base_path, 'rem')
    files = glob.glob(os.path.join(base_new_path, '*.png'))
    # cam3 = get_cam3_files(files)
    for f in tqdm(files):
        id = get_file_id(os.path.basename(f))
        # print("File Name : {} :: ID : {}".format(f, id))
        copyFile(base_orig_path, base_new_path, 'cam1_{}.png'.format(id))
        copyFile(base_orig_path, base_new_path, 'cam2_{}.png'.format(id))
        copyFile(base_orig_path, base_new_path, 'cam4_{}.png'.format(id))
        
    shutil.rmtree(base_orig_path)
    os.rename(base_new_path, base_orig_path)


if __name__ == '__main__':

    # path = '../data/raw_sim_data/train'
    # for folder in os.listdir(path):
        # cleanupFolder(os.path.join(path, folder))

    path = '../data/train'
    for file in tqdm(glob.glob(os.path.join(path, 'images','*.jpeg'))):
        maskFile = os.path.basename(file).replace('cam1_', '_mask_')
        maskFile = maskFile.replace('.jpeg','.png')
        if not os.path.exists(os.path.join(path, 'masks',maskFile)):
            os.remove(file)



