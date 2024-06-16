import random
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from glob import glob
from tqdm import tqdm
from os.path import join, split
import os
import logging


logger = logging.getLogger(__name__)

def load_metadata(params):
    # load metadata
    # if not os.path.exists(params.path):
    #     logger.error(f"El archivo de metadatos no existe: {params.path}")
    #     raise FileNotFoundError(f"El archivo de metadatos no existe: {params.path}")
    
    # Convertir SimpleNamespace a diccionario para DataFrame
    params_dict = {k: getattr(params, k) for k in params.__dict__.keys()}
    params_df = pd.DataFrame([params_dict])  # Crea un DataFrame con una fila de datos
    
    return params_df


    
    # apply splits / folds / cross-validation

