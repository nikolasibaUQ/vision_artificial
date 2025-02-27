import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats

import cv2
import os


def convert_RGB(input_path: str, output_path: str = '', save: bool = False):

    # Cargar la imagen
    img = cv2.imread(input_path)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Convertir de BGR a RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Guardar la imagen convertida
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, img_rgb)

    # Mostrar la imagen si se solicita

    cv2.imshow('Imagen RGB', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gray_image(input_path: str, output_path: str = '', save: bool = False):

    # Cargar la imagen
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if output_path != '' and save:
        output_path = os.path.abspath(output_path)
        cv2.imwrite(output_path, img)

    cv2.imshow('Imagen en escala de grises', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def binary_image(input_path: str, output_path: str = '', save: bool = False, threshold: int = 1, max_value: int = 255):

    # Cargar la imagen
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Aplicar umbralización
    _, img_binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # Guardar la imagen binarizada
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, img_binary)

    # Mostrar la imagen binarizada si se solicita
    cv2.imshow('Imagen binarizada', img_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def normal_image(input_path: str, ):

    input_path = os.path.abspath(input_path)
    img = cv2.imread(input_path)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def convert_gray_scale(input_path: str, output_path: str = '', save: bool = False):

    # Cargar la imagen
    img = cv2.imread(input_path)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Convertir de BGR a escala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Guardar la imagen en escala de grises
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, img_gray)

    # Mostrar la imagen en escala de grises si se solicita
    cv2.imshow('Imagen en escala de grises', img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convert_binary_image(input_path: str, output_path: str = '', save: bool = False, threshold: int = 128, max_value: int = 255):

    # Cargar la imagen
    img = cv2.imread(input_path)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Convertir a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Umbralizar la imagen para convertirla en blanco

    _, binary_img = cv2.threshold(
        gray_img, threshold, max_value, cv2.THRESH_BINARY)

    # Guardar la imagen binarizada
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, binary_img)

    # Mostrar la imagen binarizada si se solicita
    cv2.imshow('Imagen binarizada', binary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_image(input_path: str, output_path: str = '', save: bool = False, width: int = 400, height: int = 400):

    # Cargar la imagen
    img = cv2.imread(input_path)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Cambiar el tamaño de la imagen
    resized_img = cv2.resize(img, (width, height))

    # Guardar la imagen redimensionada
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, resized_img)

    # Mostrar la imagen redimensionada si se solicita
    cv2.imshow('Imagen redimensionada', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotate_image(input_path: str, output_path: str = '', save: bool = False, angle: int = 45):

    img = cv2.imread(input_path)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")
    # Obtener las dimensiones de la imagen
    (h, w) = img.shape[:2]

    # Establecer el centro de la imagen para rotarla
    center = (w // 2, h // 2)

    # Crear la matriz de rotación
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotar la imagen
    rotated_img = cv2.warpAffine(img, M, (w, h))

    # Guardar la imagen rotada
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, rotated_img)

    # Mostrar la imagen rotada si se solicita
    cv2.imshow('Imagen rotada', rotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def add_images(input_path1: str, input_path2: str, output_path: str = '', save: bool = False):

    # Cargar las imágenes
    img1 = cv2.imread(input_path1)
    img2 = cv2.imread(input_path2)

    # Verificar si las imágenes se cargaron correctamente
    if img1 is None or img2 is None:
        raise ValueError(
            f"Error: No se pudo cargar alguna de las imágenes. Verifica las rutas y los formatos de los archivos.")

    # Sumar las imágenes
    sum_img = cv2.add(img1, img2)

    # Guardar la imagen resultante
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, sum_img)

    # Mostrar la imagen resultante
    cv2.imshow('Suma de imágenes', sum_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def subtract_images(input_path1: str, input_path2: str, output_path: str = '', save: bool = False):

    # Cargar las imágenes
    img1 = cv2.imread(input_path1)
    img2 = cv2.imread(input_path2)

    # Verificar si las imágenes se cargaron correctamente
    if img1 is None or img2 is None:
        raise ValueError(
            f"Error: No se pudo cargar alguna de las imágenes. Verifica las rutas y los formatos de los archivos.")

    # Restar las imágenes
    diff_img = cv2.subtract(img1, img2)

    # Guardar la imagen resultante
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, diff_img)

    # Mostrar la imagen resultante
    cv2.imshow('Resta de imágenes', diff_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def multiply_images(input_path1: str, input_path2: str, output_path: str = '', save: bool = False):

    # Cargar las imágenes
    img1 = cv2.imread(input_path1)
    img2 = cv2.imread(input_path2)

    # Verificar si las imágenes se cargaron correctamente
    if img1 is None or img2 is None:
        raise ValueError(
            f"Error: No se pudo cargar alguna de las imágenes. Verifica las rutas y los formatos de los archivos.")

    # Multiplicar las imágenes
    mult_img = cv2.multiply(img1, img2)

    # Guardar la imagen resultante
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, mult_img)

    # Mostrar la imagen resultante
    cv2.imshow('Multiplicación de imágenes', mult_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def divide_images(input_path1: str, input_path2: str, output_path: str = '', save: bool = False):

    # Cargar las imágenes
    img1 = cv2.imread(input_path1)
    img2 = cv2.imread(input_path2)

    # Verificar si las imágenes se cargaron correctamente
    if img1 is None or img2 is None:
        raise ValueError(
            f"Error: No se pudo cargar alguna de las imágenes. Verifica las rutas y los formatos de los archivos.")

    # Convertir las imágenes a flotantes
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Dividir las imágenes
    div_img = cv2.divide(img1, img2)

    # Normalizar la imagen resultante
    div_img = cv2.normalize(div_img, None, 0, 255, cv2.NORM_MINMAX)

    # Convertir la imagen resultante a tipo entero
    div_img = div_img.astype(np.uint8)

    # Guardar la imagen resultante
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, div_img)

    # Mostrar la imagen resultante
    cv2.imshow('División de imágenes', div_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def and_images(input_path1: str, input_path2: str, output_path: str = '', save: bool = False):

    # Cargar las imágenes
    img1 = cv2.imread(input_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(input_path2, cv2.IMREAD_GRAYSCALE)

    # Verificar si las imágenes se cargaron correctamente
    if img1 is None or img2 is None:
        raise ValueError(
            f"Error: No se pudo cargar alguna de las imágenes. Verifica las rutas y los formatos de los archivos.")

    # Realizar la operación AND
    and_img = cv2.bitwise_and(img1, img2)

    # Guardar la imagen resultante
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, and_img)

    # Mostrar la imagen resultante
    cv2.imshow('Operación AND', and_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def or_images(input_path1: str, input_path2: str, output_path: str = '', save: bool = False):

    # Cargar las imágenes
    img1 = cv2.imread(input_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(input_path2, cv2.IMREAD_GRAYSCALE)

    # Verificar si las imágenes se cargaron correctamente
    if img1 is None or img2 is None:
        raise ValueError(
            f"Error: No se pudo cargar alguna de las imágenes. Verifica las rutas y los formatos de los archivos.")

    # Realizar la operación OR
    or_img = cv2.bitwise_or(img1, img2)

    # Guardar la imagen resultante
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, or_img)

    # Mostrar la imagen resultante
    cv2.imshow('Operación OR', or_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def blur_image(input_path: str, output_path: str = '', save: bool = False, kernel_size: int = 50):

    # Cargar la imagen
    img = cv2.imread(input_path)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Aplicar el filtro de desenfoque
    blurred_img = cv2.blur(img, (kernel_size, kernel_size))

    # Guardar la imagen filtrada
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, blurred_img)

    # Mostrar la imagen filtrada
    cv2.imshow('Desenfoque', blurred_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def gaussian_blur_image(input_path: str, output_path: str = '', save: bool = False, kernel_size: int = 51):

    # Cargar la imagen
    img = cv2.imread(input_path)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Aplicar el filtro de desenfoque gaussiano
    gaussian_blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # Guardar la imagen filtrada
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, gaussian_blurred_img)

    # Mostrar la imagen filtrada
    cv2.imshow('Desenfoque Gaussiano', gaussian_blurred_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sharpen_image(input_path: str, output_path: str = '', save: bool = False):

    # Cargar la imagen
    img = cv2.imread(input_path)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Crear un kernel para nítidez
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])

    # Aplicar el filtro de nítidez
    sharpened_img = cv2.filter2D(img, 0, sharpen_kernel)

    # Guardar la imagen filtrada
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, sharpened_img)

    # Mostrar la imagen filtrada
    cv2.imshow('Nitidez', sharpened_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def detect_edges(input_path: str, output_path: str = '', save: bool = False, low_threshold: int = 50, high_threshold: int = 150):

    # Cargar la imagen
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Aplicar el detector de bordes
    edges_img = cv2.Canny(img, low_threshold, high_threshold)

    # Guardar la imagen resultante
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, edges_img)

    # Mostrar la imagen resultante
    cv2.imshow('Detector de bordes', edges_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def emboss_image(input_path: str, output_path: str = '', save: bool = False):

    # Cargar la imagen
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Crear un kernel para el filtro de relieve
    emboss_kernel = np.array([[-2, -1, 0],
                              [-1,  1, 1],
                              [0,  1, 2]])

    # Aplicar el filtro de relieve
    embossed_img = cv2.filter2D(img, -1, emboss_kernel)

    # Guardar la imagen filtrada
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, embossed_img)

    # Mostrar la imagen filtrada
    cv2.imshow('Filtro de relieve', embossed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def dilated_image(input_path: str, output_path: str = '', save: bool = False, kernel_size: int = 5):

    # Cargar la imagen
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Crear un kernel para la dilatación
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Aplicar la dilatación
    dilated_img = cv2.dilate(img, kernel, iterations=1)

    # Guardar la imagen resultante
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, dilated_img)

    # Mostrar la imagen resultante
    cv2.imshow('Dilatación', dilated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    

   