from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.v2 as v2
import random

TEST_IMAGE = 'C:/Users/itoom/COMP3710/ADNI/AD_NC/test/AD/388206_78.jpeg'

# Pathways to relevant subfolders within ADNI folder
TEST_PATH = 'AD_NC/test/'
TRAIN_PATH = 'AD_NC/train/'
CONTROL_PATH = 'NC/'
PRESENT_PATH = 'AD/'

if __name__ == "__main__":
    pass