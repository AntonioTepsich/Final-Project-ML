import logging
import random
import torch
import numpy as np
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms

logger = logging.getLogger(__name__)

class ColorDataset(data.Dataset):
    def __init__(self, metadata):
        self.l_images = metadata['l_image']
        self.ab_images = metadata['ab_image']
        
    def __len__(self):
        return len(self.l_images)
    
    def __getitem__(self, idx):
        l_image = np.array(self.l_images[idx]).reshape((224, 224, 1))

        ab_image = np.array(self.ab_images[idx])

        l_image = transforms.ToTensor()(l_image) #cambia el rango de los valores de los píxeles de 0-255 a 0.0-1.0
        
        # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # to_tensor = transforms.ToTensor()
        # transform = transforms.Compose([
        #     to_tensor,
        #     normalize
        # ])
        # ab_image = transform(ab_image)
        
        ab_image = transforms.ToTensor()(ab_image)
        
        return l_image, ab_image