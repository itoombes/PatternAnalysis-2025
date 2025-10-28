from PIL import Image
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import re

# Regular expression to match the ADNI file name format
ADNI_PATTERN = re.compile(r"\d+_\d+.jpeg")
TEST_PATH = 'AD_NC/test/'
TRAIN_PATH = 'AD_NC/train/'
CONTROL = 'NC/'
PRESENT = 'AD/'

class Adni():
    def __init__(self, base_directory: str, validation_split: float = 0.2, validation_split_base: int | None = None):
        '''
        base_directory: Location of ADNI folder
        validation_split: Amount of dataset to be split (rounds up)
        validation_split_base: Random 
        '''
        self.directory = base_directory
        self.validation_split = validation_split
        self.validation_split_base = validation_split_base
    
    def get_test_data(self):
        '''
        Load the training dataset
        '''
        control = load_directory(self.directory+TEST_PATH+CONTROL)
        present = load_directory(self.directory+TEST_PATH+PRESENT)
        # Convert into tensors
        control = torch.from_numpy(control)
        present = torch.from_numpy(present)
        # Create markers for positive and negative
        control_out = torch.zeros(control.shape[0])
        present_out = torch.ones(present.shape[0])
        
        # Stack together
        images = torch.cat((control, present), dim=0)
        output = torch.cat((control_out, present_out), dim=0)

        # Load into dataset and return
        return TensorDataset(images, output)

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
    adni = Adni('C:/Users/itoom/COMP3710/ADNI/')
    adni.get_test_data()