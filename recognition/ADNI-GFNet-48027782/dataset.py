from PIL import Image
import os
import numpy as np
import pandas as pd
import torch
import re

import matplotlib.pyplot as plt

# Regular expression to match the ADNI file name format
ADNI_PATTERN = re.compile(r"\d+_\d+.jpeg")

def load_directory(directory) -> np.ndarray:
    '''
    Load every image within the provided directory into a
    <n_images>x240x256 numpy array

    directory: Path to directory of relevant images 
    '''
    data = list()

    if os.path.isdir(directory):
        # Iterate over every valid file in directory
        for file in os.listdir(directory):
            if ADNI_PATTERN.match(file):
                # Get patient ID from file name
                key = int(file.split('_')[0])
                # Read image, and convert into numpy array
                im = np.asarray(Image.open(directory+file))
    
                data.append(im)

    # Convert to numpy array and return
    return np.stack(data)

def load_directory_by_id(directory) -> dict[int, np.ndarray]:
    '''
    Load every image within the provided directory into a dictionary of numpy
    arrays, keyed with patient IDs. Each of these arrays is in the format
    <n_patient_images>x240x256

    directory: Path to directory of relevant images 
    '''
    id_grouped_data = dict()

    if os.path.isdir(directory):
        # Iterate over every valid file in directory
        for file in os.listdir(directory):
            if ADNI_PATTERN.match(file):
                # Get patient ID from file name
                key = int(file.split('_')[0])
                # Read image, and convert into numpy array
                im = np.asarray(Image.open(directory+file))

                # If patient ID already encountered, add to list of images
                # Else, create that list
                if key in id_grouped_data:
                    id_grouped_data[key].append(im)
                else:
                    id_grouped_data[key] = list()
                    id_grouped_data[key].append(im)
    
    # Convert lists of images to numpy arrays
    for key in id_grouped_data:
        id_grouped_data[key] = np.stack(id_grouped_data[key])

    return id_grouped_data 


if __name__ == "__main__":
    ADNI_PATH = 'C:/Users/itoom/COMP3710/ADNI/'
    TEST_PATH = ADNI_PATH+'AD_NC/test/'
    TRAIN_PATH = ADNI_PATH+'AD_NC/train/'

    print(load_directory(TEST_PATH+'AD/'))
    print(load_directory_by_id(TEST_PATH+'AD/').keys())