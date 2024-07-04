# Image Colorization
<!-- ### Introducing Convolutional KAN Networks! -->
<!-- This project extends the idea of the innovative architecture of Kolmogorov-Arnold Networks (KAN) to the Convolutional Layers, changing the classic linear transformation of the convolution to learnable non linear activations in each pixel. --> 
### Authors
This repository was made by:
 - Antonio Tepsich | atepsich@udesa.edu.ar 
 - Maximo Gubitosi | mgubitosi@udesa.edu.ar 
 <!-- - Antonio Tepsich | atepsich@udesa.edu.ar | [Twitter](https://twitter.com/antotepsich) | [LinkedIn](https://www.linkedin.com/in/antonio-tepsich/) -->
 
### Adviser
<!-- PONER LO DE TRINI -->
 - Trinidad Monreal | tmonreal@udesa.edu.ar 

## Consideraciones
Este trabajo llevo muchisimas horas de computo en la nube, es por eso que estandarizar y automatizar como se trabaja fue fundamental. Este repositorio tiene todo lo que se necesita para correr la mayorias de modelo de manera automatica.

Por razones de dependencias, la GAN que esta hecha en pytorch no entrena bien, es por eso que hicimos un notebook donde se implementa con Tensorflow y el problema se resuelve. El que deben mirar es el que esta al inicio y se llama gan_ml_definitiva.ipynb

Por el lado del modelo de denoising, hay una carpeta especifica que tiene 2 notebooks, primero se ejecuta pre_process y luego DnCNN.

Algo a tener en cuenta es por temas de espacios a compartir, no podemos subir los modelos preentrenados nuestros ya que son pesados y el campus no nos permite compartirlos. Si es necesario lo podemos pasar por otro medio. A su vez los path estan hecho de manera local, cuando usabamos colab conectada con la VM de Google Cloud, asi que habria que poner los path locales tuyos.

# Installation
```bash
git clone git@github.com/AntonioTepsich/Final-Project-ML
cd Final_Project_ML
pip install -r requirements.txt
```

Ademas deben crear los archivos logs, runs, results en caso de ser la primera vez que corra.

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
En caso de querer otro dataset, buscar en la carpeta scrips



### 2- Entrenar modelos
Especificar las configuraciones de los modelos a entrenar y correr el siguiente comando:
```bash
./go.sh
```

### 3- TensorBoard

```bash
tensorboard --logdir runs
```
