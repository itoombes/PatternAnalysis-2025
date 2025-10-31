from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
from torchvision.io.image import decode_image
import random

# Location of data folder
# Make sure to include the last '/' character
ADNI_ROOT = '/home/groups/comp3710/ADNI/'

# Pre-processing transforms used on the data
TRANSFORM = v2.Compose([
    v2.ToDtype(torch.float32, scale=True), # Force datatype compatibility
])

# Pathways to relevant subfolders within ADNI folder
TEST_PATH = 'AD_NC/test/'
TRAIN_PATH = 'AD_NC/train/'
CONTROL_PATH = 'NC/'
PRESENT_PATH = 'AD/'

def split_validation_and_training(folder: str, seed: int | None = None,
                                  validation_split: float = 0.2) -> tuple[dict, dict]:
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
                                   int(len(training_map) * validation_split))
    
    # Remove pid entries from training_map and place into validation_map
    validation_map = dict()
    for pid in validation_ids:
        validation_map[pid] = training_map.pop(pid) 

    return (validation_map, training_map)

class ImageDataset(Dataset):
    def __init__(self, subfolder: str, cn_files: list, ad_files: list,
                 transform: v2.Transform | None = None):
        '''
        subfolder: Either TEST_PATH or TRAIN_PATH, depending on where images are
            being stored. 
        cn_files: Files in the 'Cognitive Normal' category
        ad_files: Files in the 'Alzheimer's Detected' category
        transform: torchvision transform to apply to image
        '''
        self.subfolder = subfolder
        self.cn_files = cn_files
        self.ad_files = ad_files
        self.transform = transform
    
    def __len__(self) -> int:
        '''
        Treats ad_files and cn_files as though they are a combined list
        '''
        return len(self.cn_files) + len(self.ad_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        '''
        Retrieve image & label from dataset
        Treats the file names as though they are a combined list, beginning with
        cn_files.
        '''
        label = None
        filepath = ADNI_ROOT + self.subfolder
        # Extract label and full filepath
        if (idx >= len(self.cn_files)):
            # Using the ad_files list
            label = 1
            # Index is offset by the length of cn_files
            filename = self.ad_files[idx - len(self.cn_files)] 
            filepath += PRESENT_PATH + filename 
        else:
            # Using the cn_files list
            label = 0
            filepath += CONTROL_PATH + self.cn_files[idx]
        
        image = decode_image(filepath)
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_test_dataset(transform: v2.Transform | None = None) -> ImageDataset:
    '''
    Load the dataset containing the test data

    transform: Image transform to be applied to each image
    '''
    data_root = ADNI_ROOT + TEST_PATH 
    # Get file names within each subfolder
    cn_files = os.listdir(data_root + CONTROL_PATH)
    ad_files = os.listdir(data_root + PRESENT_PATH) 

    # Initialise and return the dataset
    return ImageDataset(TEST_PATH, cn_files, ad_files, transform=transform)

def get_test_dataloader(**kwargs) -> DataLoader:
    '''
    Create a dataloader containing the test data.

    **kwargs: Keyword arguments passed to dataloader
    '''
    # Load the test dataset with default transform
    dataset = get_test_dataset(transform = TRANSFORM)
    return DataLoader(dataset, **kwargs)

def get_validation_and_training_datasets(validation_split: float = 0.2,
                                        transform: v2.Transform | None = None,
                                        seed: int | None = None) -> tuple[Dataset, Dataset]:
    '''
    Read the training folder, separate it into a validation and test dataset.

    validation_split: Proportion of dataset to put in validation dataset
    transform: Image transform to be applied to each image
    seed: Seed to initialise random.seed() for reproducibility
    '''
    train_root = ADNI_ROOT + TRAIN_PATH
    cn_root = train_root + CONTROL_PATH
    ad_root = train_root + PRESENT_PATH

    # Read the dataset files and split into testing & validation sets
    validation_cn, training_cn = split_validation_and_training(cn_root, 
        seed = seed, validation_split = validation_split)
    validation_ad, training_ad = split_validation_and_training(ad_root, 
        seed = seed, validation_split = validation_split)
    # Convert into lists of lists of file names
    validation_cn = list(validation_cn.values())
    validation_ad = list(validation_ad.values())
    training_cn = list(training_cn.values())
    training_ad = list(training_ad.values())
    # Flatten
    validation_cn = [c for r in validation_cn for c in r]
    validation_ad = [c for r in validation_ad for c in r]
    training_cn = [c for r in training_cn for c in r]
    training_ad = [c for r in training_ad for c in r]

    # Convert into datasets & return
    validation_set = ImageDataset(TRAIN_PATH, validation_cn, validation_ad,
                                  transform = transform)
    training_set = ImageDataset(TRAIN_PATH, training_cn, training_ad,
                                transform = transform)
    return (validation_set, training_set)

def get_validation_and_training_dataloaders(validation_batch_size: int,
        training_batch_size: int, validation_split: float = 0.2,
        shuffle_training: bool = True, shuffle_validation: bool = False,
        seed: int | None = None) -> tuple[DataLoader, DataLoader]:
    '''
    Create dataloaders containing the training data, separated into training
    and validation sets. Returns in order (validation, training).

    validation_batch_size: Batch size of validation data loader
    training_batch_size: Batch size of training data loader
    validation_split: Proportion of ADNI training dataset to use for validation
    shuffle_training: Whether to shuffle the training loader
    shuffle_validation: Whether to shuffle the validation loader
    seed: Seed passed to random.seed(), for reproducibility
    '''
    validation_set, training_set = get_validation_and_training_datasets(
        validation_split=validation_split, transform=TRANSFORM, seed = seed)
    
    validation = DataLoader(validation_set, batch_size = validation_batch_size,
                            shuffle = shuffle_validation)
    training = DataLoader(training_set, batch_size = training_batch_size,
                          shuffle = shuffle_training)
    
    return (validation, training)
