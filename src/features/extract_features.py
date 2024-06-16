import torch
import logging
import importlib
import numpy as np
from os.path import isfile, isdir, join, sep
from os import makedirs
from tqdm import tqdm

logger = logging.getLogger(__name__)

def extract_features(metadata, params):
    feature_module = importlib.import_module(f'src.features.{params.feature}')
    metadata = getattr(feature_module, params.feature)(metadata, params)
    metadata = normalization(metadata, params)
    return metadata
        
def normalization(metadata, params):
    if params.norm == 'lab':
        metadata = lab_normalization(metadata, params)
    # elif params.norm == 'speaker':
    #     metadata = speaker_normalization(metadata, params)
    else:
        raise ValueError(f'Normalization method {params.norm} not implemented')
    return metadata

def lab_normalization(metadata, params):
    metadata['l_image'] = metadata['l_image'] / 255.0
    metadata['ab_image'] = metadata['ab_image'] / 127.5 - 1.0
    return metadata
   
def speaker_normalization(metadata, params):
    pass

