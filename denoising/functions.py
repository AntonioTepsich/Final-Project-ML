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
    # Open image
    image = Image.open(input_path).convert("L")  # convert to grayscale
    
    # obtain the original size of the image
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    
    # calculate the new size of the image
    if aspect_ratio > 1:
        # image wider than tall
        new_width = size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        # image taller than wide
        new_height = size[1]
        new_width = int(new_height * aspect_ratio)
    
    # resize the image
    # if the original image is smaller than the target size, use LANCZOS interpolation, otherwise use BILINEAR
    if original_width < size[0] or original_height < size[1]:
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    else:
        resized_image = image.resize((new_width, new_height), Image.BILINEAR)
    
    # create a new image with the target size and paste the resized image in the center
    new_image = Image.new("L", size)
    new_image.paste(resized_image, ((size[0] - new_width) // 2, (size[1] - new_height) // 2))
    
    return new_image



def add_noise_to_image(image_array, noise_level=0.03, seed=42):
    """
    Adds Gaussian noise to an image.
    
    Parameters:
        image_array (numpy.array): Image to add noise to.
        noise_level (float): Standard deviation of the Gaussian noise to add to the image.
        seed (int): Seed for the random number generator.
    
    Returns:
        numpy.array: Noisy image.
    """
    rng = RandomState(seed)
    mean = 0
    std = noise_level * 255  # scale the noise level to the range [0, 255]
    
    # create Gaussian noise with the same shape
    gauss = rng.normal(mean, std, image_array.shape)
    
    # add the noise to the image
    noisy_image = image_array + gauss
    noisy_image = np.clip(noisy_image, 0, 255)  # Insure that the values are within the valid range for uint8
    
    return noisy_image.astype(np.uint8)

