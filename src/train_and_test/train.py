import logging 
import torch
import time
import numpy as np
from src.callbacks.Writer import TensorBoardWriter
from os.path import join, split, sep
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, accuracy_score
import os
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from src.utils.useful_functions import lab_to_rgb


logger = logging.getLogger(__name__)

def display_progress(cond, real, fake, current_epoch = 0, figsize=(20,15)):
    """
    Save cond, real (original) and generated (fake)
    images in one panel 
    """
    cond = cond.detach().cpu().permute(1, 2, 0)   
    real = real.detach().cpu().permute(1, 2, 0)
    fake = fake.detach().cpu().permute(1, 2, 0)
    
    images = [cond, real, fake]
    titles = ['input','real','generated']
    print(f'Epoch: {current_epoch}')
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for idx,img in enumerate(images):
        if idx == 0:
            ab = torch.zeros((224,224,2))
            img = torch.cat([images[0]* 100, ab], dim=2).numpy()
            imgan = lab2rgb(img)
        else:
            imgan = lab_to_rgb(images[0],img)
        ax[idx].imshow(imgan)
        ax[idx].axis("off")
    for idx, title in enumerate(titles):    
        ax[idx].set_title('{}'.format(title))
    plt.show()

def train(model, train_loader, val_loader, early_stopper, params, context):
    if params.tboard:
        writer = TensorBoardWriter(context['save_path'])

    running_train_losses = []
    running_validation_losses = []


    # Checkea si hay un modelo pre-entrenado y sino arranca de 0
    if params.pre_trained_model is not None or os.path.exists(join(context['save_path'], 'full_model.pt')):
        if params.pre_trained_model is None:
            checkpoint = torch.load(join(context['save_path'],'full_model.pt'))
        else:
            checkpoint = torch.load(join(context['save_path'],params.pre_trained_model))
            
        model.load_state_dict(checkpoint["model_state_dict"])
        # model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        initial_time = checkpoint["time"]
        running_train_losses = checkpoint["running_train_losses"]
        running_validation_losses = checkpoint["running_validation_losses"]
    else:
        initial_time = 0
    
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # train loop and validation loop
    start = time.time()
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
        running_train_losses.append(total_loss)
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
        running_validation_losses.append(total_loss)

        if (epoch % 10) == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                # "optimizer_state_dict": model.optimizer.state_dict(),
                "time": time.time() - start + initial_time,
                "running_train_losses": running_train_losses,
                "running_validation_losses": running_validation_losses
            }
            checkpoint_path = join(context['save_path'], f"checkpoint_{epoch}.pt")
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch:03} with val loss {val_loss:.4f}")

            display_progress(data[0][0], data[1][0], model.predict(data[0])[0], current_epoch=epoch)

        # _continue = False if early_stopper.early_stop(val_loss) else True
        _continue = True
        # ...
        
        # callbacks calls
        # ...

        writer.update_status(epoch, train_loss, val_loss)    
        if epoch == params.max_epochs:
            # if params.types == 'GAN':
            #     full_model = {
            #         "epoch": epoch,
            #         "model_state_dict": model.generator.state_dict(),
            #         "optimizer_state_dict": model.optimizer_G.state_dict(),
            #         "time": time.time() - start + initial_time,
            #         "running_train_losses": running_train_losses,
            #         "running_validation_losses": running_validation_losses
            #     }
            # else:
            full_model = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                # "optimizer_state_dict": model.optimizer.state_dict(),
                "time": time.time() - start + initial_time,
                "running_train_losses": running_train_losses,
                "running_validation_losses": running_validation_losses
            }
            full_model_path = join(context['save_path'], f'full_model.pt')
            torch.save(full_model, full_model_path)
            logger.info("Max epochs reached! Stopping training..")
            _continue = False
        if not _continue:
            return
        

