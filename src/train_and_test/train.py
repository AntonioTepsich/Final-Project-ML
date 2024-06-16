import logging 
import torch
import time
import numpy as np
from src.callbacks.Writer import TensorBoardWriter
from os.path import join, split, sep
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, accuracy_score

logger = logging.getLogger(__name__)

def train(model, train_loader, val_loader, early_stopper, params, context):
    if params.tboard:
        writer = TensorBoardWriter(context['save_path'])
    prev_val_loss = float("inf")
    best_val_loss = float("inf")
    
    for epoch in range(1, params.max_epochs+1):
        model.train()
        total_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        # train loop
        for i, data in pbar:
            loss = model.train_step(data)
            pbar.set_description(f"Epoch {epoch}. Train loss: {loss:.4f}")
            total_loss += loss
        train_loss = total_loss / (i+1)
        # ...
        
        model.eval()
        total_loss = 0
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        # validation loop
        with torch.no_grad():
            for i, data in pbar:
                loss = model.eval_step(data)
                pbar.set_description(f"Epoch {epoch}. Val loss: {loss:.4f}")
                total_loss += loss
        val_loss = total_loss / (i+1)
        # _continue = False if early_stopper.early_stop(val_loss) else True
        _continue = True
        # ...
        
        # callbacks calls
        # ...

        writer.update_status(epoch, train_loss, val_loss)
        if (val_loss+0.001) < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"Saving the temp best epoch at {epoch:03} with val loss {val_loss:.4f}")
            best_file_path = join(context['save_path'], 'temp_best_epoch.pth')
            torch.save(model, best_file_path)
            
        if epoch == params.max_epochs:
            # creo que falta guardar el modelo full entrenado
            #guardar modelo final .pt
            full_model_path = join(context['save_path'], 'final_model.pt')
            torch.save(model, full_model_path)
            logger.info("Max epochs reached! Stopping training..")
            _continue = False
        if not _continue:
            return
        

