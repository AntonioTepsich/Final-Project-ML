import numpy as np
import torch
from skimage.color import lab2rgb
import warnings
import matplotlib.pyplot as plt

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


def mostrar(train_loader):
    """
    Show a batch of images from the train_loader
    """
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