# Convolutional Kolmogorov-Arnold Network (CKAN) 
### Introducing Convolutional KAN Networks!
This project extends the idea of the innovative architecture of Kolmogorov-Arnold Networks (KAN) to the Convolutional Layers, changing the classic linear transformation of the convolution to learnable non linear activations in each pixel. 
### Authors
This repository was made by:
 - Antonio Tepsich | atepsich@udesa.edu.ar 
 - Maximo Gubitosi | mgubitosi@udesa.edu.ar 
 <!-- - Antonio Tepsich | atepsich@udesa.edu.ar | [Twitter](https://twitter.com/antotepsich) | [LinkedIn](https://www.linkedin.com/in/antonio-tepsich/) -->
 
### Adviser
<!-- PONER LO DE TRINI -->
 - Antonio Tepsich | atepsich@udesa.edu.ar 


### Credits
This repository is based on the following papers:
<!-- PONER LISTADO DE PAPERS -->
 - [here](https://github.com/Blealtan/efficient-kan).


# Installation
```bash
git clone git@github.com/AntonioTepsich/Final-Project-ML
cd Final_Project_ML
pip install -r requirements.txt
```

# Usage
### 1-Descargar el Dataset de Kaggle
Deberas ingresar tu usuario y apiKey de Kaggle luego de ejecutar el siguiente comando:
```bash
python scripts/download_datasets.py
```

Ejemplo:
```bash
Your Kaggle username: XXXX
Your Kaggle Key: XXXXXXXXXXXXXXXXXX
```

### 2- Entrenar modelos
Especificar las configuraciones de los modelos a entrenar y correr el siguiente comando:
```bash
./go.sh
```

### 3- TensorBoard

```bash
tensorboard --logdir runs
```
