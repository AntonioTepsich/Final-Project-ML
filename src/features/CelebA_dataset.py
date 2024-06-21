import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils import data

class RGBToGrayLAB:
    def __call__(self, img):
        # Transformación de la imagen a espacio de color LAB
        # Debe retornar los componentes L y AB
        # Placeholder para el proceso real
        img_lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2Lab)
        l, a, b = cv2.split(img_lab)
        return l, np.stack([a, b], axis=-1)

class CelebADataset(data.Dataset):
    def __init__(self, image_folder, reduced=False, limit=500, transform=None, target_transform=None):
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
        l_image = torch.tensor(l_image, dtype=torch.float32).unsqueeze(0)
        ab_image = torch.tensor(ab_image, dtype=torch.float32).permute(2, 0, 1)
        if self.target_transform:
            ab_image = self.target_transform(ab_image)
        return l_image, ab_image

def load_celeba_images(image_folder, limit=None):
    dataset = CelebADataset(image_folder, reduced=True, limit=limit)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    l_images = []
    ab_images = []
    
    for l_image, ab_image in loader:
        l_images.append(l_image.squeeze(0).numpy())
        ab_images.append(ab_image.permute(1, 2, 0).numpy())

    l_images = np.stack(l_images)
    ab_images = np.concatenate(ab_images)
    
    data = {'l_image': l_images, 'ab_image': ab_images}
    return data

# # Uso del código
# image_folder = '/path/to/celeba/images'  # Actualiza con la ruta correcta al dataset
# data = load_celeba_images(image_folder, limit=25000)