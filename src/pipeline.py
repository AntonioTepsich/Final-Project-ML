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

from src.utils.useful_functions import lab_to_rgb
from src.utils.useful_functions import mostrar

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
