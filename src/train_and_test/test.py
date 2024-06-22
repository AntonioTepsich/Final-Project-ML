import logging 
import numpy as np
import torch
import json
from os.path import join, split, sep
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, accuracy_score
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from torch import cat
from torch import nn
import os
from src.utils.useful_functions import lab_to_rgb




logger = logging.getLogger(__name__)


def test(test_loader, context, model, pre_trained_model):
    
    # Checkea si hay un modelo pre-entrenado y sino usa el ultimo
    if pre_trained_model is None:
        best_model_path = join(context['save_path'], 'full_model.pt')
    else:
        best_model_path = join(context['save_path'], pre_trained_model)

    model_data = torch.load(best_model_path)
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    total_loss = 0
    save_images = True
    max_images_to_save = 2
    images_saved = 0

    if save_images:
        directory = join(context['save_path'], 'images')
        if not os.path.exists(directory):
            os.makedirs(directory)

    buffer_images = [[], [], []]  # To store images for saving in groups of 5

    for j, (gray, color) in enumerate(test_loader):
        outputs = model.predict(gray).detach()
        cond = gray.detach().cpu().permute(0, 2, 3, 1)
        real = color.detach().cpu().permute(0, 2, 3, 1)
        fake = outputs.detach().cpu().permute(0, 2, 3, 1)
        images = [cond, real, fake]

        batch_size = cond.shape[0]
        for i in range(batch_size):
            buffer_images[0].append(images[0][i])
            buffer_images[1].append(images[1][i])
            buffer_images[2].append(images[2][i])

            if len(buffer_images[0]) == 5 and images_saved < max_images_to_save:
                fig, axes = plt.subplots(3, 5, figsize=(16, 10))

                for k in range(5):
                    ab = torch.zeros((224, 224, 2))
                    input_img = torch.cat([buffer_images[0][k] * 100, ab], dim=2).numpy()
                    input_img = lab2rgb(input_img)
                    real_img = lab_to_rgb(buffer_images[0][k], buffer_images[1][k])
                    fake_img = lab_to_rgb(buffer_images[0][k], buffer_images[2][k])

                    axes[0, k].imshow(input_img)
                    axes[0, k].axis('off')
                    if k == 2:
                        axes[0, k].set_title("INPUT", fontsize=20)

                    axes[1, k].imshow(real_img)
                    axes[1, k].axis('off')
                    if k == 2:
                        axes[1, k].set_title("REAL", fontsize=20)

                    axes[2, k].imshow(fake_img)
                    axes[2, k].axis('off')
                    if k == 2:
                        axes[2, k].set_title("PREDICTED", fontsize=20)

                plt.savefig(join(directory, f"test_{images_saved}.png"))
                plt.close(fig)
                images_saved += 1

                buffer_images = [[], [], []]  # Clear buffer

        with torch.no_grad():
            mse = torch.nn.MSELoss()
            for i in range(batch_size):
                fake_img = lab_to_rgb(images[0][i], images[2][i])
                real_img = lab_to_rgb(images[0][i], images[1][i])
                fake_img_tensor = torch.from_numpy(fake_img).permute(2, 0, 1)
                real_img_tensor = torch.from_numpy(real_img).permute(2, 0, 1)
                loss = mse(fake_img_tensor, real_img_tensor)
                total_loss += loss.item()

    logger.info(f"Test Loss: {total_loss:.5f}")

    with open(join(context['save_path'], "testing_total_loss.json"), "w") as results_file:
        json.dump({"total_loss": total_loss}, results_file)

    return