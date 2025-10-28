from PIL import Image
import os
import numpy as np
import pandas as pd
import re

def get_patient_ids(directory: str) -> set: 
    '''
    Returns the unique patient ids within the specified folder of the ADNI
    dataset.

    Assumes files are in the format "<patient id>_<image number>.png"

    directory: Path to directory of relevant images 
    '''
    ids = set()

    # Regular expression to match the ADNI file name format
    adni_pattern = re.compile(r"\d+_\d+.jpeg")


    if os.path.isdir(directory):
        # Iterate over every file in directory
        for file in os.listdir(directory):
            # Add the patient ID of any valid image to the set
            if adni_pattern.match(file):
                ids.add(file.split('_')[0])

    return ids 
