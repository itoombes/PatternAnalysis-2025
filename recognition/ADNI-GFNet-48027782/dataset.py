from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.v2 as v2
import random

TEST_IMAGE = 'C:/Users/itoom/COMP3710/ADNI/AD_NC/test/AD/388206_78.jpeg'
ADNI_ROOT = 'C:/Users/itoom/COMP3710/ADNI/'

# Pathways to relevant subfolders within ADNI folder
TEST_PATH = 'AD_NC/test/'
TRAIN_PATH = 'AD_NC/train/'
CONTROL_PATH = 'NC/'
PRESENT_PATH = 'AD/'

def split_validation_and_training(folder: str, seed: int | None = None,
                                  split_amount: float = 0.2) -> tuple[dict, dict]:
    '''
    Separate the filenames based by patient ID into a dict.
    Then randomly separates this dict into validation and training dicts.

    Returns in order (validation_map, training_map)
    
    folder: Folder containing the images to split (i.e. ADNI/AD_NC/train/NC/)
    seed: Seed to initialise random.seed() for reproducibility
    split_amount: Proportion of data to move into validation set
    '''
    # Map patient id (pid) to associated file names
    training_map = dict()
    for file in os.listdir(folder):
        # Extract pid from file name
        pid = int(file.split('_')[0])
        # Add file name to dictionary
        if pid in training_map.keys():
            training_map[pid].append(file)
        else:
            training_map[pid] = [file,]

    # Set random split seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Randomly select a proportion of pid's to use for validation set
    validation_ids = random.sample(list(training_map.keys()),
                                   int(len(training_map) * split_amount))
    
    # Remove pid entries from training_map and place into validation_map
    validation_map = dict()
    for pid in validation_ids:
        validation_map[pid] = training_map.pop(pid) 

    return (validation_map, training_map)

if __name__ == "__main__":
    validation, training = split_validation_and_training(ADNI_ROOT+TEST_PATH+CONTROL_PATH)
    print(validation.keys())
    print(training.keys())