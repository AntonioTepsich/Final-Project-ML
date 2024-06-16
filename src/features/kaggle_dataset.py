import logging
import librosa
import numpy as np
from os.path import join, isfile, sep
from os import makedirs
import os
from tqdm import tqdm
import torch

logger = logging.getLogger(__name__)

def kaggle_dataset(metadata, params):
    # instantiate opensmile

    # extract features
    
    l_images = np.load(os.path.join(metadata.path_l.values[0], 'gray_scale.npy'))
    ab_images = np.concatenate([
        np.load(os.path.join(metadata.path_ab.values[0], 'ab1.npy')),                   # Tiene 10.000 imagenes
        np.load(os.path.join(metadata.path_ab.values[0], 'ab2.npy')),                   # Tiene 10.000 imagenes
        np.load(os.path.join(metadata.path_ab.values[0], 'ab3.npy'))                    # Tiene 5.000 imagenes
    ])
    
    if metadata.limit is not None:
        l_images = l_images[:metadata.limit.values[0]]
        ab_images = ab_images[:metadata.limit.values[0]]
        

    data= {'l_image': l_images, 'ab_image': ab_images}
    return data