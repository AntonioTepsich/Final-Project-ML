import logging
import random
import torch
import numpy as np
from tqdm import tqdm
from torch.utils import data

logger = logging.getLogger(__name__)

class ColorDataset(data.Dataset):
    def __init__(self, metadata):
        self.l_images = metadata['l_image']
        self.ab_images = metadata['ab_image']
        
    def __len__(self):
        return len(self.l_images)
    
    def __getitem__(self, idx):
        l_image = self.l_images[idx]
        ab_image = self.ab_images[idx]
        # l_image = torch.tensor(l_image, dtype=torch.float32).unsqueeze(0) / 255.0  # de 0 y 255 a [0,1]
        l_image = torch.tensor(l_image, dtype=torch.float32).unsqueeze(0)  # de 0 y 255
        # ab_image = torch.tensor(ab_image, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0 # de 0 y 255 a [-1,1]
        ab_image = torch.tensor(ab_image, dtype=torch.float32).permute(2, 0, 1) # de 0 y 255

        return l_image, ab_image