import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ImageColorizationDataset(Dataset):
    def __init__(self, l_dir, ab_dir, reduced=False, limit=3000, transform=None, target_transform=None):
        """
        :param l_dir: Directory where the L channel numpy file is stored.
        :param ab_dir: Directory where the A and B channel numpy files are stored.
        :param reduced: If True, reduces the dataset size to 3000 images.
        :param transform: (Optional) Transform to be applied on the L channel images.
        :param target_transform: (Optional) Transform to be applied on the A and B channel images.
        """
        self.l_images = np.load(os.path.join(l_dir, 'gray_scale.npy'))  # Tiene 25.000 imagenes
        self.ab_images = np.concatenate([
            np.load(os.path.join(ab_dir, 'ab1.npy')),                   # Tiene 10.000 imagenes
            np.load(os.path.join(ab_dir, 'ab2.npy')),                   # Tiene 10.000 imagenes
            np.load(os.path.join(ab_dir, 'ab3.npy'))                    # Tiene 5.000 imagenes
        ])
        
        if reduced:
            self.l_images = self.l_images[:limit]
            self.ab_images = self.ab_images[:limit]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.l_images)

    def __getitem__(self, idx):
        l_image = self.l_images[idx]
        ab_image = self.ab_images[idx]

        # Normalization and conversion to tensors should be done according to your specific needs
        l_image = torch.tensor(l_image, dtype=torch.float32).unsqueeze(0) / 255.0  # de 0 y 255 a [0,1]
        ab_image = torch.tensor(ab_image, dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0 # de 0 y 255 a [-1,1]

        if self.transform:
            l_image = self.transform(l_image)
        if self.target_transform:
            ab_image = self.target_transform(ab_image)

        return l_image, ab_image