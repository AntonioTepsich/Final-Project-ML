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


logger = logging.getLogger(__name__)


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def print_stats(arr, name):
    print(f"{name} - min: {arr.min()}, max: {arr.max()}, mean: {arr.mean()}, std: {arr.std()}")



def test(test_loader, context):
    best_model_path = join(context['save_path'], 'temp_best_epoch.pth')
    model = torch.load(best_model_path)
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

        outputs = model(gray)

        if save_images:
            fig = plt.figure(figsize=(8, 8))

        for i in range(1, columns + 1):
            if save_images:
                fig.add_subplot(rows, columns, i)
            
            gray_image = tensor_to_numpy(gray[i-1]).squeeze()
            img = gray_image

            if i == 3:
                plt.title("Input")

            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
            if save_images:
                plt.imshow(img, cmap="gray")

        for i in range(1, columns + 1):
            if save_images:
                fig.add_subplot(rows, columns, i + columns)
            if i == 3:
                plt.title("Actual")

            gray_image = tensor_to_numpy(gray[i-1]).squeeze()
            ab_image = tensor_to_numpy(color[i-1])
            img_lab = np.zeros((224, 224, 3), dtype=np.float32)
            img_lab[:,:,0] = gray_image * 100
            img_lab[:,:,1:] = (ab_image.transpose(1, 2, 0)) * 127.5
            img_lab = np.clip(img_lab, 0, 100)  # Clip values

            img = lab2rgb(img_lab)

            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
            if save_images:
                plt.imshow(img)

        with torch.no_grad():
            for i in range(1, columns + 1):
                if save_images:
                    fig.add_subplot(rows, columns, i + 2 * columns)
                mse = nn.MSELoss()

                gray_image = tensor_to_numpy(gray[i-1]).squeeze()
                ab_image_predicted = tensor_to_numpy(outputs[i-1])

                # Desnormalización correcta de la salida del modelo
                ab_image_predicted = (ab_image_predicted + 1) * 127.5  # Escalar de [-1, 1] a [0, 255]

                img_lab_predicted = np.zeros((224, 224, 3), dtype=np.float32)
                img_lab_predicted[:, :, 0] = np.clip(gray_image * 100, 0, 100)  # L en el rango [0, 100]
                img_lab_predicted[:, :, 1:] = ab_image_predicted.transpose(1, 2, 0)  # AB en el rango [-128, 127]

                img = lab2rgb(img_lab_predicted)

                ab_image = tensor_to_numpy(color[i-1])
                img_lab = np.zeros((224, 224, 3), dtype=np.float32)
                img_lab[:,:,0] = np.clip(gray_image * 100, 0, 100)  # Clip values
                img_lab[:,:,1:] = np.clip((ab_image.transpose(1, 2, 0)) * 127, -128, 127)  # Clip values

                img_true = lab2rgb(img_lab)

                img_tensor = torch.from_numpy(img).permute(2, 0, 1)
                img_true_tensor = torch.from_numpy(img_true).permute(2, 0, 1)

                loss = mse(img_tensor, img_true_tensor)

                total_loss += loss.item()
                plt.gca().get_yaxis().set_visible(False)
                plt.gca().set_xticks([])
                plt.gca().set_xlabel(f"{loss.item():.5f}")

                if i == 3:
                    plt.title("Predicted")

                if save_images:
                    plt.imshow(img)

        if save_images:
            plt.savefig(join(directory, f"test_{j}.png"))
            plt.close()

        j += 1

    logger.info(f"Test Loss: {total_loss}")
    
    with open(join(context['save_path'],"testing_total_loss.json"), "w") as results_file:
        json.dump({"total_loss": total_loss}, results_file)

    return