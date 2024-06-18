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
import warnings




logger = logging.getLogger(__name__)


def test(test_loader, context, model, pre_trained_model):
    if pre_trained_model is None:
        best_model_path = join(context['save_path'], 'full_model.pt')
        
    else:
        best_model_path = join(context['save_path'], pre_trained_model)

    model_data = torch.load(best_model_path)
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    columns = 5
    rows = 3

    data_iterator = iter(test_loader)

    total_loss = 0
    save_images = True
    if save_images:
        directory = join(context['save_path'], 'images')
        if not os.path.exists(directory):
            os.makedirs(directory)

    j = 0
    while True:
        try:
            gray, color = next(data_iterator)
        except StopIteration:
            break

        outputs = model(gray).detach()

        cond = gray
        real = color
        fake = outputs

        cond = cond.detach().cpu().permute(0,2,3,1)
        real = real.detach().cpu().permute(0,2,3,1)
        fake = fake.detach().cpu().permute(0,2,3,1)

        images = [cond, real, fake]

        if save_images:
            fig = plt.figure(figsize=(16, 10))

        for i in range(1, columns + 1):
            if save_images:
                fig.add_subplot(rows, columns, i)
            ab = torch.zeros((224, 224, 2))
            img = torch.cat([images[0][i-1]*100,ab],dim=2).numpy()
            imgan = lab2rgb(img)

            if i == 3:
                plt.title("INPUT", fontsize=20)

            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)

            if save_images:
                plt.imshow(imgan)

        for i in range(1, columns + 1):
            if save_images:
                fig.add_subplot(rows, columns, i + columns)
            if i == 3:
                plt.title("REAL", fontsize=20)

            imgan = lab_to_rgb(images[0][i-1], images[1][i-1])

            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
            if save_images:
                plt.imshow(imgan)

        with torch.no_grad():
            for i in range(1, columns + 1):
                if save_images:
                    fig.add_subplot(rows, columns, i + 2 * columns)
                mse = nn.MSELoss()

                img = lab_to_rgb(images[0][i-1], images[2][i-1])

                img_true = lab_to_rgb(images[0][i-1], images[1][i-1])


                #esto puede estar mal
                img_tensor = torch.from_numpy(img).permute(2, 0, 1)
                img_true_tensor = torch.from_numpy(img_true).permute(2, 0, 1)

                loss = mse(img_tensor, img_true_tensor)

                total_loss += loss.item()
                plt.gca().get_yaxis().set_visible(False)
                plt.gca().set_xticks([])
                plt.gca().set_xlabel(f"{loss.item():.5f}", fontsize=14)

                if i == 3:
                    plt.title("PREDICTED", fontsize=20)

                if save_images:
                    plt.imshow(img)

        if save_images:
            plt.savefig(join(directory, f"test_{j}.png"))
            plt.close()
        
        j += 1
    
    logger.info(f"Test Loss: {total_loss:.5f}")

    with open(join(context['save_path'],"testing_total_loss.json"), "w") as results_file:
        json.dump({"total_loss": total_loss}, results_file)

    return

def lab_to_rgb(L, ab):
    """
    Takes an image or a batch of images and converts from LAB space to RGB
    """
    L = np.clip(L  * 100,0,100)
    ab = np.clip((ab - 0.5) * 128 * 2,-128,127)
    Lab = torch.cat([L, ab], dim=2).numpy()
    rgb_imgs = []
    for img in Lab:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)