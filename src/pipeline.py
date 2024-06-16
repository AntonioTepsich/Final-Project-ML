import logging
import importlib
import os, random
import torch
from os.path import join, split, exists

# from src.dataloaders.AudioDataloader import AudioDataloader
from src.dataloaders.ColorDataset import ColorDataset

from src.utils.files import create_result_folder
from src.callbacks.EarlyStopper import EarlyStopper
from src.datasets.load_metadata import load_metadata
from src.features.extract_features import extract_features

from torch.utils.data import DataLoader, random_split
from src.train_and_test.train import train
from src.train_and_test.test import test

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import lab2rgb

logger = logging.getLogger(__name__)


def run_experiment(model_params, data_params, features_params):
    """
    Este ejemplo es para un caso en el cual solamente entrenemos y evaluemos un modelo
    basado en redes. La idea sera modificar estos pasos para que se ajusten a las necesidades
    de lo que cada uno vaya a proponer como modelo. Podria mantenerse esto mismo y simplemente
    complejizar las subfunciones, o bien modificar esta funcion para que se ajuste a un flujo de
    trabajo distinto, eliminando o agregando pasos.
    """
    context = {}
    # create result folder
    context['save_path'] = create_result_folder(model_params, data_params, features_params)
    
    # get metadata
    metadata = load_metadata(data_params)
    metadata.to_pickle(os.path.join(context['save_path'],'metadata.pkl'))
    
    # extract features
    features = extract_features(metadata, features_params)
    
    # dataloaders
    train_loader, val_loader, test_loader = get_dataloader(features, model_params.dataloader_params)
    
    mostrar(train_loader)

    # # load model
    # model_params.input_dim = features_params.dim
    model = load_model(model_params)
    
    # # train
    # # early_stopper = EarlyStopper(patience=model_params.early_stop_patience)
    early_stopper = False
    model = train(model, train_loader, val_loader, early_stopper, model_params, context)
    
    
    # test
    test(test_loader, context)
    logger.info('Experiment completed')
    return 






def load_model(params):
    model_module = importlib.import_module(f'src.models.{params.name}')
    model = getattr(model_module, params.name)(params)
    model = model.to(params.device)
    return model



def get_dataloader(features, params):
    dataset = ColorDataset(features)
    # dataloader = AudioDataloader(dataset, **params)

    # Calcular las longitudes de cada conjunto
    batch_size = params['batch_size']
    num_workers = params['num_workers']
    train_ratio = params['train_ratio']
    valid_ratio = params['valid_ratio']
    shuffle_train = params['shuffle_train']
    shuffle_valid = params['shuffle_valid']
    shuffle_test = params['shuffle_test']

    total_count = len(dataset)
    train_count = int(total_count * train_ratio)
    valid_count = int(total_count * valid_ratio)
    test_count = total_count - train_count - valid_count  # Asegura que sumen el total

    # Dividir el dataset
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_count, valid_count, test_count])

    # Crear DataLoader para cada conjunto
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_train)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_valid)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_test)
    
    return train_loader, valid_loader, test_loader



def mostrar(train_loader):
    def tensor_to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    dataiter = iter(train_loader)
    images, labels = next(dataiter)  # Carga un batch del DataLoader

    # Configuración del plot
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))  # 4 filas, 4 columnas para 8 pares de imágenes

    for i in range(8):  # Solo necesitamos 8 pares de imágenes, total 16 subplots
        row = i // 2  # Cada fila tiene 2 pares
        col = (i % 2) * 2  # Columna alterna para B&W y Colored (0, 2 para B&W; 1, 3 para Colored)

        l_image = tensor_to_numpy(images[i]).squeeze()  # [H, W], quita el canal si es 1
        ab_image = tensor_to_numpy(labels[i])  # [2, H, W]

        # Asegurarse de que los datos están en la escala correcta
        l_image = l_image * 100  # Escalar L de [0, 1] a [0, 100]
        ab_image = ab_image * 128  # Escalar a y b de [-1, 1] a [-128, 127]

        # Imagen en escala de grises
        axs[row, col].imshow(l_image, cmap='gray')
        axs[row, col].set_title(f'B&W - Id: {i}', fontsize=10)
        axs[row, col].axis('off')  # Desactiva los ejes

        # Imagen coloreada
        img_lab = np.zeros((224, 224, 3), dtype=np.float32)
        img_lab[:,:,0] = l_image  # L canal
        img_lab[:,:,1:] = ab_image.transpose(1, 2, 0)  # a y b canales
        img_rgb = lab2rgb(img_lab)  # Convierte LAB a RGB
        axs[row, col + 1].imshow(img_rgb)
        axs[row, col + 1].set_title(f'Colored - Id: {i}', fontsize=10)
        axs[row, col + 1].axis('off')

    plt.tight_layout()
    plt.show()