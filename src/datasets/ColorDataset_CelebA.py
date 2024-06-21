import logging
import random
import torch
import os 
import numpy as np
from tqdm import tqdm
from torch.utils import data
from PIL import Image
from skimage.color import rgb2lab
from torchvision import transforms

logger = logging.getLogger(__name__)

class RGBToGrayLAB:
    def __call__(self, img):
        img = np.array(img)
        lab = rgb2lab(img).astype("float32")
        l = lab[..., 0] / 100.0  # Escalar L de [0, 100] a [0, 1]
        ab = lab[..., 1:] / 128.0  # Escalar a y b de [-128, 127] a [-1, 1]
        return l, ab

    def __repr__(self):
        return self.__class__.__name__ + '()'
    

class CelebADataset(data.Dataset):
    def __init__(self, image_folder, reduced=False, limit=500, transform=None, target_transform=None):
        """
        :param image_folder: Path to the folder containing the images.
        :param reduced: If True, reduces the dataset size to the limit specified.
        :param transform: (Optional) Transform to be applied on the L channel images.
        :param target_transform: (Optional) Transform to be applied on the A and B channel images.
        """
        self.image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.jpg')]
        
        if reduced:
            self.image_paths = self.image_paths[:limit]

        self.transform = transform
        self.target_transform = target_transform
        self.rgb_to_graylab = RGBToGrayLAB()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        l_image, ab_image = self.rgb_to_graylab(img)

        # Convert to tensors
        l_image = torch.tensor(l_image, dtype=torch.float32).unsqueeze(0)  # Añadir la dimensión del canal
        ab_image = torch.tensor(ab_image, dtype=torch.float32).permute(2, 0, 1)  # Reorganizar dimensiones a (C, H, W)

        if self.target_transform:
            ab_image = self.target_transform(ab_image)

        return l_image, ab_image