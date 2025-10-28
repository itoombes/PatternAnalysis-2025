from PIL import Image
import os
import numpy as np
import pandas as pd
import torch
import re

import matplotlib.pyplot as plt

# Regular expression to match the ADNI file name format
ADNI_PATTERN = re.compile(r"\d+_\d+.jpeg")

def get_patient_ids(directory: str) -> set: 
    '''
    Returns the unique patient ids within the specified folder of the ADNI
    dataset.

    Assumes files are in the format "<patient id>_<image number>.png"

    directory: Path to directory of relevant images 
    '''
    ids = set()

    if os.path.isdir(directory):
        # Iterate over every file in directory
        for file in os.listdir(directory):
            # Add the patient ID of any valid image to the set
            if ADNI_PATTERN.match(file):
                ids.add(file.split('_')[0])

    return ids 

def load_image(path: str) -> np.ndarray | None:
    """
    Load an image into a 2d tensor. Returns None if invalid path.
    
    path: image to load
    """
    if (os.path.exists(path)):
        im = Image.open(path)
        return np.asarray(im)

if __name__ == "__main__":
    ADNI_PATH = 'C:/Users/itoom/COMP3710/ADNI/'
    TEST_PATH = ADNI_PATH+'AD_NC/test/'
    TRAIN_PATH = ADNI_PATH+'AD_NC/train/'

    # 30520 240 x 256 images
    images = np.zeros((30520, 240, 256)) 
    i = 0

    for path in (TEST_PATH+'AD/', TEST_PATH+'NC/', TRAIN_PATH+'AD/', TRAIN_PATH+'NC/'):
        for i, file in enumerate(os.listdir(path)):
            im = load_image(path+file)
            if im is not None:
                images[i] = im
                i += 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    images = torch.from_numpy(images)
    images.to(device)
    print("Determining maximum...")
    image_max = torch.max(images, dim=0).values

    plt.imshow(image_max.cpu().numpy(), cmap='gray')
    plt.show()