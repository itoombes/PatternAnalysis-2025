from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import re
import random

# Regular expression to match the ADNI file name format
ADNI_PATTERN = re.compile(r"\d+_\d+.jpeg")
# Pathways to relevant subfolders within ADNI folder
TEST_PATH = 'AD_NC/test/'
TRAIN_PATH = 'AD_NC/train/'
CONTROL_PATH = 'NC/'
PRESENT_PATH = 'AD/'

class Adni():
    def __init__(self, base_directory: str, validation_split: float = 0.2,
                 validation_split_base: int | None = None):
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
    
    def get_training_and_validation_data(self) -> tuple[TensorDataset, TensorDataset]:
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
            # Get the patient data and remove it from the dictionary
            new_data = control.pop(pid)
            # Convert to tensor and append
            validation_control.append(torch.from_numpy(new_data))

        validation_present = list()
        for pid in validation_present_ids:
            # Get the patient data and remove it from the dictionary
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

        validation = TensorDataset(validation_images, validation_out)

        # Convert the remaining data in the dictionaries into combined tensors
        training_control = list()
        training_present = list()
        for v in control.values():
            training_control.append(torch.from_numpy(v))
        for v in present.values():
            training_present.append(torch.from_numpy(v))
        training_control = torch.cat(training_control)
        training_present = torch.cat(training_present)

        # Convert into TensorDataset
        training_control_out = torch.zeros(training_control.shape[0])
        training_present_out = torch.ones(training_present.shape[0])
        
        training_images = torch.cat((training_control, training_present))
        training_out = torch.cat((training_control_out, training_present_out))
        training = TensorDataset(training_images, training_out)

        return (training, validation)
    
    def get_test_dataloader(self, **kwargs) -> DataLoader:
        '''
        Return dataloader for testing set.

        **kwargs: Arguments passed to DataLoader constructor
        '''
        testset = self.get_test_data()
        return DataLoader(testset, **kwargs)
    
    def get_training_and_validation_dataloaders(self, training_batch_size: int,
            validate_batch_size: int, shuffle: bool = True,
            **kwargs) -> tuple[DataLoader, DataLoader]:
        '''
        Returns DataLoaders for training and validation set, in that order.
        
        training_batch_size: Batch size of training data
        validate_batch_size: Batch size of validation data
        shuffle: Whether to shuffle the datasets
        **kwargs: Keyword arguments to pass to constructors
        '''

        trainset, validateset = self.get_training_and_validation_data()
        train_loader = DataLoader(trainset, batch_size=training_batch_size,
                shuffle=shuffle, **kwargs)

        validate_loader = DataLoader(validateset,
                batch_size=validate_batch_size, shuffle=shuffle, **kwargs)

        return (train_loader, validate_loader) 

def load_image(filepath: str) -> np.ndarray:
    '''
    Load an image and return in a numpy format which is compatible with torch.
    (i.e., channels x height x width)

    filepath: Path to image
    '''
    im = np.asarray(Image.open(filepath))[np.newaxis, :, :]
    return im

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
                # Read image
                im = load_image(directory+file)

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
                im = load_image(directory+file)

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