import librosa
import logging
import random
import torch
import time
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

logger = logging.getLogger(__name__)

# --------------MODIFICAR----------------
class ColorDataloader():
    def __init__(self, dataset, batch_size, num_workers, train_ratio, valid_ratio, shuffle_train, shuffle_valid, shuffle_test):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.shuffle_train = shuffle_train
        self.shuffle_valid = shuffle_valid
        self.shuffle_test = shuffle_test

    def split_dataset(self):
        total_count = len(self.dataset)
        train_count = int(total_count * self.train_ratio)
        valid_count = int(total_count * self.valid_ratio)
        test_count = total_count - train_count - valid_count
        return random_split(self.dataset, [train_count, valid_count, test_count])

    def create_dataloaders(self):
        train_dataset, valid_dataset, test_dataset = self.split_dataset()
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle_train, pin_memory = True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle_valid, pin_memory = True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle_test, pin_memory = True)
        return train_loader, valid_loader, test_loader
