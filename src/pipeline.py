import logging
import importlib
import os
import torch

# from src.dataloaders.AudioDataloader import AudioDataloader
from src.dataloaders.ColorDataloader import ColorDataloader
from src.datasets.ColorDataset import ColorDataset
from src.datasets.load_metadata import load_metadata

from src.utils.files import create_result_folder
from src.callbacks.EarlyStopper import EarlyStopper
from src.features.extract_features import extract_features

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
    
    # mostrar(train_loader)

    # load model
    model = load_model(model_params)
    
    # train
    # early_stopper = EarlyStopper(patience=model_params.early_stop_patience)
    early_stopper = False
    train(model, train_loader, val_loader, early_stopper, model_params, context)
    
    
    # test
    test(test_loader, context, model, model_params.pre_trained_model)
    logger.info('Experiment completed')
    return 




def load_model(params):
    model_module = importlib.import_module(f'src.models.{params.name}')
    model = getattr(model_module, params.name)(params)
    model = model.to(params.device)
    return model



def get_dataloader(features, params):
    dataset = ColorDataset(features)
    dataloader = ColorDataloader(dataset, **params)
    train_loader, valid_loader, test_loader = dataloader.create_dataloaders()

    return train_loader, valid_loader, test_loader


def lab_to_rgb(L, ab):
    """
    Takes an image or a batch of images and converts from LAB space to RGB
    """
    L = L  * 100
    ab = (ab - 0.5) * 128 * 2
    Lab = torch.cat([L, ab], dim=2).numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def mostrar(train_loader):
    dataiter = iter(train_loader)
    images, labels = next(dataiter)  # Carga un batch del DataLoader

    for i in range(2):  # Solo necesitamos 8 pares de imágenes, total 16 subplots

        cond = labels[i]
        real = images[i]

        cond = cond.detach().cpu().permute(1,2,0)
        real = real.detach().cpu().permute(1,2,0)
        
        
        
        fotos = [real,cond]
        titles = ['real', 'input']
        fig,ax = plt.subplots(1 ,2 ,figsize=(20,15))
        for idx,img in enumerate(fotos):
            if idx == 0:
                ab = torch.zeros((224, 224, 2))
                img = torch.cat([fotos[0]*100,ab],dim=2).numpy()
                imgan = lab2rgb(img)
            else:
                imgan = lab_to_rgb(fotos[0],img)
            ax[idx].imshow(imgan)
            ax[idx].axis('off')
        for idx, title in enumerate(titles):
            ax[idx].set_title('{}'.format(title))
        plt.show()