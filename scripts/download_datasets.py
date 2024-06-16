import opendatasets as od
import os

od.download("https://www.kaggle.com/datasets/shravankumar9892/image-colorization","./")

def rename_folder(old_name, new_name):
    try:
        os.rename(old_name, new_name)
        print(f"La carpeta '{new_name}' se ha creado con éxito.")
    except OSError as e:
        print(f"Error: {e}")


old_folder_name = "image-colorization"
new_folder_name = "data"
rename_folder(old_folder_name, new_folder_name)