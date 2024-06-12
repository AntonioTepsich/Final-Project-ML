from src.dataset import ImageColorizationDataset
from src.training import train_model
from src.models import u_net_v1 


from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

import zipfile
import os


extracted_folder = 'data'
if not os.path.exists(extracted_folder):
    with zipfile.ZipFile("archive.zip","r") as zip_ref:
        zip_ref.extractall(extracted_folder)

# List all the files in the extracted folder
os.listdir(extracted_folder)


ab_path = Path("data/ab/ab")
l_path = Path("data/l")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

batch_size = 32
# Prepare the Datasets
all_dataset = ImageColorizationDataset(l_dir=l_path, ab_dir=ab_path, reduced=True)

# Definir las proporciones para entrenamiento, validación y prueba
train_ratio = 0.75
valid_ratio = 0.15
test_ratio = 0.1

# Calcular las longitudes de cada conjunto
total_count = len(all_dataset)
train_count = int(total_count * train_ratio)
valid_count = int(total_count * valid_ratio)
test_count = total_count - train_count - valid_count  # Asegura que sumen el total

# Establecer la semilla para reproducibilidad
torch.manual_seed(42)

# Dividir el dataset
train_dataset, valid_dataset, test_dataset = random_split(all_dataset, [train_count, valid_count, test_count])

# Crear DataLoader para cada conjunto
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


architecture = u_net_v1.UNet_1()
file_name = "unet_1.0"
train_model(train_loader, architecture, file_name, epochs=10, learning_rate=0.001, device=device)