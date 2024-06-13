import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage.color import rgb2lab

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



class RGBToGrayLAB:
    def __call__(self, img):
        img = np.array(img)
        lab = rgb2lab(img).astype("float32")
        l = lab[..., 0] / 100.0  # Escalar L de [0, 100] a [0, 1]
        ab = lab[..., 1:] / 128.0  # Escalar a y b de [-128, 127] a [-1, 1]
        return l, ab

    def __repr__(self):
        return self.__class__.__name__ + '()'

class CelebADataset(Dataset):
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
