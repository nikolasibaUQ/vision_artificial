import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import stats



def get_media_image(input_path: str):

    # Cargar la imagen
    img = cv2.imread(input_path)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Obtener la mediana de la imagen
    pixels = img.flatten()
    media = np.mean(pixels)

    return media

def get_mode_image(input_path: str):

    # Cargar la imagen
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Obtener la moda de la imagen
    pixels = img.flatten()
    mode = stats.mode(pixels , keepdims=True )[0][0]

    return mode



def get_mean_image(input_path: str):

    # Cargar la imagen
    img = cv2.imread(input_path)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    pixels = img.flatten()
    # Obtener la media de la imagen
    mean = np.mean(pixels)

    return mean


def get_desviation_image(input_path: str):

    # Cargar la imagen
    img = cv2.imread(input_path)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    pixels = img.flatten()
    desviation = np.std(pixels)

    return desviation
    

   