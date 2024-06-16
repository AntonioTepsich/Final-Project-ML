import logging 
import torch
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from os.path import join, split, sep
from datetime import datetime

class TensorBoardWriter(object):
    def __init__(self, save_path):
        time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        tboard_path = join('runs', sep.join(save_path.split(sep)[1:]), str(time))
        self.writer = SummaryWriter(log_dir=tboard_path)
    
    def update_status(self, epoch, train_loss, val_loss):
        self.writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)