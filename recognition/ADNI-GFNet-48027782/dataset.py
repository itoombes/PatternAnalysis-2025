from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
import re
import random

# Regular expression to match the ADNI file name format
ADNI_PATTERN = re.compile(r"\d+_\d+.jpeg")
# Locations of 
TEST_PATH = 'AD_NC/test/'
TRAIN_PATH = 'AD_NC/train/'
CONTROL_PATH = 'NC/'
PRESENT_PATH = 'AD/'

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

        # Ensure validation split is in range. Otherwise, choose 0.2
        if not (validation_split > 0 and validation_split < 1):
            self.validation_split = 0.2
    
    def get_test_data(self) -> TensorDataset:
        '''
        Load the training dataset
        '''
        control = load_directory(self.directory+TEST_PATH+CONTROL_PATH)
        present = load_directory(self.directory+TEST_PATH+PRESENT_PATH)
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
    
    def get_training_and_validation_data(self):
        '''
        Loads the training dataset, and segments it into a training and
        validation set. Returns in order (training, validation). 
        '''
        control = load_directory_by_id(self.directory+TRAIN_PATH+CONTROL_PATH)
        present = load_directory_by_id(self.directory+TRAIN_PATH+PRESENT_PATH)

        # Ensure repeatable between calls
        if self.validation_split_base is not None:
            random.seed(self.validation_split_base)

        # Randomly select IDs of patients to include in validaiton set
        validation_control_ids = random.sample(list(control.keys()),
                int(len(control.keys()) * self.validation_split))
        validation_present_ids = random.sample(list(present.keys()),
                int(len(present.keys()) * self.validation_split))
        
        # Create validation groups
        validation_control = list()
        for pid in validation_control_ids:
            # Get the patient data and remove it from the existing set
            new_data = control.pop(pid)
            # Convert to tensor and append
            validation_control.append(torch.from_numpy(new_data))

        validation_present = list()
        for pid in validation_present_ids:
            # Get teh data and remove it from the existing set
            new_data = present.pop(pid)
            # Convert to tensor and append
            validation_present.append(torch.from_numpy(new_data))
        
        # Convert validation into TensorDataset
        validation_control = torch.cat(validation_control)
        validation_present = torch.cat(validation_present)
        validation_control_out = torch.zeros(validation_control.shape[0])
        validation_present_out = torch.ones(validation_present.shape[0])

        validation_images = torch.cat((validation_control, validation_present))
        validation_out = torch.cat((validation_control_out, validation_present_out))

        print(validation_images.shape)
        print(validation_out.shape)
        validation = TensorDataset(validation_images, validation_out)



def load_directory(directory: str) -> np.ndarray:
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

def load_directory_by_id(directory: str) -> dict[int, np.ndarray]:
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
    adni.get_training_and_validation_data()