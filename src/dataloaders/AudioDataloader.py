import librosa
import logging
import random
import torch
import time
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils import data
from src.dataloaders.ColorDataset import AudioDataset

logger = logging.getLogger(__name__)

# --------------MODIFICAR----------------
class AudioDataloader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataloader, self).__init__(*args, **kwargs)
        # self.collate_fn = collate_fn
        # Calcular las longitudes de cada conjunto
        total_count = len(all_dataset)
        train_count = int(total_count * train_ratio)
        valid_count = int(total_count * valid_ratio)
        test_count = total_count - train_count - valid_count  # Asegura que sumen el total

        # Establecer la semilla para reproducibilidad
        torch.manual_seed(42)

        # Dividir el dataset
        train_dataset, valid_dataset, test_dataset = random_split(all_dataset, [train_count, valid_count, test_count])

        # Crear DataLoader para cada conjunto
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def collate_fn(baches):
    # read preprocessed features or 
    # compute features on-the-fly
    pass
        
