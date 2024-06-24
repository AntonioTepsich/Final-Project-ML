import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils import data

def rgb_to_graylab(img):
    # Transformación de la imagen a espacio de color LAB
    img = img.resize((224, 224))  # Redimensionar la imagen a 224x224 antes de convertirla a LAB
    img_lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(img_lab)
    # print("Dimensiones de L, A, B después de cv2.split:", l.shape, a.shape, b.shape)
    return l, np.stack([a, b], axis=-1)

def CelebA_dataset(metadata, params):
    limit = metadata.limit.values[0]
    image_paths = [os.path.join(metadata.path.values[0], fname) for fname in os.listdir(metadata.path.values[0]) if fname.endswith('.jpg')]
    if limit is not None:
        image_paths = image_paths[:limit]
    
    print("Número de rutas de imágenes procesadas:", len(image_paths))

    l_images = []  # Lista para almacenar todas las imágenes L
    ab_images = []  # Lista para almacenar todas las imágenes AB

    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        l_image, ab_image = rgb_to_graylab(img)
        l_images.append(l_image)
        ab_images.append(ab_image)

    data = {'l_image': l_images, 'ab_image': ab_images}  # Diccionario con listas de arrays
    return data
# # Uso del código
# image_folder = '/path/to/celeba/images'  # Actualiza con la ruta correcta al dataset
# data = load_celeba_images(image_folder, limit=25000)