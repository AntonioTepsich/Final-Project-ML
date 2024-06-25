import opendatasets as od
import os

def rename_folder(old_name, new_name):
    try:
        os.rename(old_name, new_name)
        print(f"La carpeta '{new_name}' se ha creado con éxito.")
    except OSError as e:
        print(f"Error: {e}")

# Celeba dataset
od.download("https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data","./")

old_folder_name = "celeba-dataset"
new_folder_name = "CelebA"
rename_folder(old_folder_name, new_folder_name)