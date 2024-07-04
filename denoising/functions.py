import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
from numpy.random import RandomState
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.models import load_model



def resize_image(input_path, size=(256, 256)):
    # Abrir la imagen
    image = Image.open(input_path).convert("L")  # Convertir a escala de grises
    
    # Obtener tamaño original
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    
    # Calcular nuevo tamaño
    if aspect_ratio > 1:
        # Imagen más ancha que alta
        new_width = size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        # Imagen más alta que ancha
        new_height = size[1]
        new_width = int(new_height * aspect_ratio)
    
    # Redimensionar la imagen manteniendo la relación de aspecto
    # si la imagen es mas chica que el tamaño deseado, uso interpolación LANCZOS, sino LINEAL
    if original_width < size[0] or original_height < size[1]:
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    else:
        resized_image = image.resize((new_width, new_height), Image.BILINEAR)
    
    # Crear un nuevo fondo de 224x224 y pegar la imagen redimensionada en el centro
    new_image = Image.new("L", size)
    new_image.paste(resized_image, ((size[0] - new_width) // 2, (size[1] - new_height) // 2))
    
    return new_image



def add_noise_to_image(image_array, noise_level=0.03, seed=42):
    """
    Agrega ruido gaussiano a una imagen.
    
    Parámetros:
        image_array (numpy.array): Imagen de entrada en formato de array NumPy.
        noise_level (float): Nivel de ruido como fracción de la intensidad máxima de la imagen (0-1).
        seed (int): Seed para la generación de números aleatorios, para reproducibilidad.
    
    Retorna:
        numpy.array: Imagen con ruido añadido.
    """
    rng = RandomState(seed)
    mean = 0
    std = noise_level * 255  # Escalar el nivel de ruido por la intensidad máxima de un pixel
    
    # Crear ruido gaussiano
    gauss = rng.normal(mean, std, image_array.shape)
    
    # Agregar ruido a la imagen original
    noisy_image = image_array + gauss
    noisy_image = np.clip(noisy_image, 0, 255)  # Asegurarse de que los valores estén dentro del rango válido para uint8
    
    return noisy_image.astype(np.uint8)

