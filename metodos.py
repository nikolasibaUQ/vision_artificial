import cv2
import matplotlib.pyplot as plt
import numpy as np


import cv2
import os


def getImage(input_path: str):
    """Obtiene la ruta absoluta de una imagen dada su ruta relativa o absoluta.

    Args:
        input_path (str): Ruta de la imagen.

    Returns:
        img: 
    """
    # Convertir a ruta absoluta

    return cv2.imread(os.path.abspath(input_path))


def convert_RGB(input_path: str, output_path: str = '', save: bool = False, show: bool = False):

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

    if show:
        cv2.imshow('Imagen RGB', img_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def gray_image(input_path: str, output_path: str = '', save: bool = False, show: bool = False):

    # Cargar la imagen
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if output_path != '' and save:
        output_path = os.path.abspath(output_path)
        cv2.imwrite(output_path, img)

    if show:
        cv2.imshow('Imagen en escala de grises', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def binary_image(input_path: str, output_path: str = '', save: bool = False, threshold: int = 1, max_value: int = 255, show: bool = False):

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
    if show:
        cv2.imshow('Imagen binarizada', img_binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def normal_image(input_path: str, show: bool = False):

    input_path = os.path.abspath(input_path)
    img = cv2.imread(input_path)

    if show:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()


def convert_gray_scale(input_path: str, output_path: str = '', save: bool = False, show: bool = False):

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
    if show:
        cv2.imshow('Imagen en escala de grises', img_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def convert_binary_image(input_path: str, output_path: str = '', save: bool = False, threshold: int = 128, max_value: int = 255, show: bool = False):

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
    if show:
        cv2.imshow('Imagen binarizada', binary_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def resize_image(input_path: str, output_path: str = '', save: bool = False, width: int = 400, height: int = 400, show: bool = False):

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
    if show:
        cv2.imshow('Imagen redimensionada', resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def rotate_image(input_path: str, output_path: str = '', save: bool = False, angle: int = 45, show: bool = False):

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
    if show:
        cv2.imshow('Imagen rotada', rotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def add_images(input_path1: str, input_path2: str, output_path: str = '', save: bool = False, show: bool = False):

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
    if show:
        cv2.imshow('Suma de imágenes', sum_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def subtract_images(input_path1: str, input_path2: str, output_path: str = '', save: bool = False, show: bool = False):

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
    if show:
        cv2.imshow('Resta de imágenes', diff_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def multiply_images(input_path1: str, input_path2: str, output_path: str = '', save: bool = False, show: bool = False):

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
    if show:
        cv2.imshow('Multiplicación de imágenes', mult_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def divide_images(input_path1: str, input_path2: str, output_path: str = '', save: bool = False, show: bool = False):

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
    if show:
        cv2.imshow('División de imágenes', div_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def and_images(input_path1: str, input_path2: str, output_path: str = '', save: bool = False, show: bool = False):

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
    if show:
        cv2.imshow('Operación AND', and_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def or_images(input_path1: str, input_path2: str, output_path: str = '', save: bool = False, show: bool = False):

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
    if show:
        cv2.imshow('Operación OR', or_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def not_images(input_path: str, output_path: str = '', save: bool = False, show: bool = False):

    # Cargar la imagen
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Realizar la operación NOT
    not_img = cv2.bitwise_not(img)

    # Guardar la imagen resultante
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, not_img)

    # Mostrar la imagen resultante
    if show:
        cv2.imshow('Operación NOT', not_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def blur_image(input_path: str, output_path: str = '', save: bool = False, kernel_size: int = 50, show: bool = False):

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
    if show:
        cv2.imshow('Desenfoque', blurred_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def gaussian_blur_image(input_path: str, output_path: str = '', save: bool = False, kernel_size: int = 51, show: bool = False):

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
    if show:
        cv2.imshow('Desenfoque Gaussiano', gaussian_blurred_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def sharpen_image(input_path: str, output_path: str = '', save: bool = False, show: bool = False):

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
    if show:
        cv2.imshow('Nitidez', sharpened_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_edges(input_path: str, output_path: str = '', save: bool = False, low_threshold: int = 50, high_threshold: int = 150, show: bool = False):

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
    if show:
        cv2.imshow('Detector de bordes', edges_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def emboss_image(input_path: str, output_path: str = '', save: bool = False, show: bool = False):

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
    if show:
        cv2.imshow('Filtro de relieve', embossed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def dilated_image(input_path: str, output_path: str = '', save: bool = False, kernel_size: int = 5, show: bool = False):

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
    if show:
        cv2.imshow('Dilatación', dilated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def delete_noides_opened_image(input_path: str, output_path: str = '', save: bool = False, kernel_size: int = 5, show: bool = False):

    # Cargar la imagen
    img = cv2.imread(input_path, 0)

    # Verificar si la imagen se cargó correctamente

    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Crear un kernel para la apertura
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    opening_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Guardar la imagen resultante
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, opening_img)

    if show:
        cv2.imshow('Apertura', opening_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def delete_noides_closed_image(input_path: str, output_path: str = '', save: bool = False, kernel_size: int = 5, show: bool = False):

    # Cargar la imagen
    img = cv2.imread(input_path, 0)

    # Verificar si la imagen se cargó correctamente

    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Crear un kernel para la apertura
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Guardar la imagen resultante
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, closed_img)

    if show:
        cv2.imshow('Cerradura', closed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def gradient_image(input_path: str, output_path: str = '', save: bool = False, kernel_size: int = 5, show: bool = False):

    # Cargar la imagen
    img = cv2.imread(input_path, 0)

    # Verificar si la imagen se cargó correctamente

    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Crear un kernel para la apertura
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    # Guardar la imagen resultante
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, gradient)

    if show:
        cv2.imshow('Gradiente', gradient)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def fill_noise_opened(input_path: str, output_path: str = '', save: bool = False, kernel_size: int = 5, show: bool = False):

    # Cargar la imagen
    img = cv2.imread(input_path, 0)

    # Verificar si la imagen se cargó correctamente

    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Crear un kernel para la apertura
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Crear un kernel para la apertura
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    closed_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel)

    # Guardar la imagen resultante
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, closed_img)

    if show:
        cv2.imshow('Ruido eliminado', closed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def fill_noise_closed(input_path: str, output_path: str = '', save: bool = False, kernel_size: int = 5, show: bool = False):
    # Cargar la imagen
    img = cv2.imread(input_path, 0)

    # Verificar si la imagen se cargó correctamente

    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Crear un kernel para la apertura
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Guardar la imagen resultante
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, closed_img)

    if show:
        cv2.imshow('Ruido eliminado', closed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def segmentation_umbral_image(input_path: str, output_path: str = '', save: bool = False, min_umbral: int = 128, max_umbral: int = 255, show: bool = False):

    # Cargar la imagen
    img = cv2.imread(input_path, 0)

    # Verificar si la imagen se cargó correctamente

    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Crear un kernel para la apertura
    # se puso 200 ya que mas bajo algunos elementos se pierden
    ret, thresholded_image = cv2.threshold(
        img, min_umbral, max_umbral, cv2.THRESH_BINARY)

    # Guardar la imagen resultante
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, thresholded_image)

    # Mostrar la imagen resultante
    if show:
        cv2.imshow('Segmentación por umbral', thresholded_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def segmentation_adaptative_image(input_path: str, output_path: str = '', save: bool = False, max_umbral: int = 255, show: bool = False):

    # Cargar la imagen
    img = cv2.imread(input_path, 0)

    # Verificar si la imagen se cargó correctamente

    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Crear un kernel para la apertura
    # se puso 200 ya que mas bajo algunos elementos se pierden
    thresholded_image = cv2.adaptiveThreshold(
        img, max_umbral, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Guardar la imagen resultante
    if output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, thresholded_image)

    # Mostrar la imagen resultante
    if show:

        cv2.imshow('Segmentación adaptativa', thresholded_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def segmentation_contornos(input_path: str, output_path: str = '', save: bool = False, min_umbral: int = 128, max_umbral: int = 255, show: bool = False):

    img = cv2.imread(input_path, 0)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Aplicar umbralización
    _, thresholded_image = cv2.threshold(
        img, min_umbral, max_umbral, cv2.THRESH_BINARY)

    # Encontrar los contornos
    contours, _ = cv2.findContours(
        thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convertir la imagen binarizada a color para dibujar los contornos
    img_contours = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

    # Guardar la imagen resultante si se requiere
    if save and output_path:
        output_path = os.path.abspath(output_path)  # Convertir a ruta absoluta
        cv2.imwrite(output_path, img_contours)

    # Mostrar la imagen con contornos usando Matplotlib

    if show:
        plt.figure(figsize=(6, 6))
        plt.title('Segmentación por Contornos')
        # Convertir BGR a RGB para mostrar en Matplotlib
        plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Ocultar los ejes
        plt.show()


def segmentation_kmeans(input_path: str, output_path: str = '', save: bool = False, k: int = 2, show: bool = False):

    # Cargar imagen en color
    image = cv2.imread(input_path)

    # Verificar si la imagen se cargó correctamente
    if image is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Convertir la imagen en una matriz de datos 2D (cada fila representa un píxel con 3 valores RGB)
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)  # Convertir a flotante

    # Definir criterios de k-means: Máx 100 iteraciones o precisión mínima de 0.2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Aplicar k-means
    ret, label, center = cv2.kmeans(
        Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convertir los centros a enteros de 8 bits
    center = np.uint8(center)

    # Reemplazar los píxeles por el centro de su respectivo cluster
    res = center[label.flatten()]
    result_image = res.reshape((image.shape))

    # Guardar la imagen segmentada si es necesario
    if save and output_path:
        output_path = os.path.abspath(output_path)  # Convertir a ruta absoluta
        cv2.imwrite(output_path, result_image)

    # Mostrar la imagen segmentada con Matplotlib
    if show:
        plt.figure(figsize=(6, 6))
        plt.title(f'Segmentación con K-means (k={k})')
        # Convertir BGR a RGB para Matplotlib
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Ocultar ejes
        plt.show()


def segmentation_watershed(input_path: str, output_path: str = '', save: bool = False, show: bool = False):

    # Cargar la imagen en color
    image = cv2.imread(input_path)

    # Verificar si la imagen se cargó correctamente
    if image is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresholded_image = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Hallar los contornos
    dist_transform = cv2.distanceTransform(thresholded_image, cv2.DIST_L2, 5)
    _, markers = cv2.threshold(
        dist_transform, 0.7*dist_transform.max(), 255, 0)

    # Aplicar watershed
    markers = np.int32(markers)
    cv2.watershed(image, markers)

    # Marcamos los bordes en color rojo
    image[markers == -1] = [0, 0, 255]

    # Mostrar la imagen segmentada
    # cv2.imshow('Segmentación con Watershed', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if show:
        plt.title('Segmentación con Watershed')
        plt.imshow(image, cmap='gray')
        plt.axis('off')  # Ocultar los ejes
        plt.show()


def region_growing(image, seed_points, threshold=20):
    """Algoritmo de crecimiento de regiones a partir de puntos semilla.

    Args:
        image (numpy.ndarray): Imagen en escala de grises.
        seed_points (list): Lista de tuplas (fila, columna).
        threshold (int): Umbral para el crecimiento de la región.

    Returns:
        numpy.ndarray: Máscara resultante con la región segmentada.
    """
    rows, cols = image.shape
    mask = np.zeros_like(image, dtype=np.uint8)  # Máscara inicial

    # Filtrar puntos semilla fuera de los límites
    valid_seeds = []
    for seed in seed_points:
        row, col = seed
        if 0 <= row < rows and 0 <= col < cols:
            valid_seeds.append(seed)
        else:
            print(
                f"El punto semilla {seed} está fuera de los límites de la imagen (rows: {rows}, cols: {cols}). Se omite.")

    for seed in valid_seeds:
        row, col = seed
        mask[row, col] = 255  # Marcar la semilla
        region_mean = image[row, col]  # Valor inicial de la semilla

        # Expansión de la región: recorrer la imagen y agregar píxeles que cumplan con el umbral
        for i in range(rows):
            for j in range(cols):
                if mask[i, j] == 0:  # Si el píxel aún no está marcado
                    if abs(int(image[i, j]) - int(region_mean)) < threshold:
                        mask[i, j] = 255

    return mask


def segmentation_region_growing(input_path: str, output_path: str = '', save: bool = False, show: bool = False):
    """Segmentación de una imagen usando crecimiento de regiones."""

    # Cargar la imagen
    image = cv2.imread(input_path)

    # Verificar si la imagen se cargó correctamente
    if image is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo."
        )

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Definir puntos semilla (ajustar según la imagen)
    seed_points = [(100, 100), (200, 200)]
    result_mask = region_growing(gray, seed_points, threshold=20)

    # Mostrar la imagen original y la segmentación
    fig, ax = plt.subplots(2, 1, figsize=(6, 10))

    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Imagen Original")
    ax[0].axis("off")

    if show:
        plt.show()
        plt.close(fig)
    ax[1].axis("off")

    if show:
        plt.show()

    # Guardar la imagen si se especifica
    if save and output_path != '':
        cv2.imwrite(output_path, result_mask)


def segmentation_watershed_contours(input_path: str, output_path: str = '', save: bool = False, show: bool = False):
    """Segmentación de una imagen usando Watershed basado en contornos."""

    # Cargar la imagen
    img = cv2.imread(input_path)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo."
        )

    # Convertir la imagen a escala de grises
   # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar un umbral binario
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontrar los contornos
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una imagen vacía para los marcadores con 1 canal
    # Changed to gray and dtype to int32
    markers = np.zeros_like(gray, dtype=np.int32)

    # Definir los seed points (por ejemplo, marcar manualmente las regiones de interés)
    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i + 1), -1)

    # Marcar el fondo como -1
    markers[thresh == 0] = -1

    # Aplicar Watershed
    cv2.watershed(img, markers)

    # Las fronteras de los objetos serán marcadas con -1
    img[markers == -1] = [0, 0, 255]  # Rojo para las fronteras

    # Mostrar la imagen resultante

    plt.title('Segmentación Watershed')
    plt.imshow(img)
    plt.axis('off')  # Ocultar los ejes
    if show:
        plt.show()

    # Guardar la imagen si se especifica
    if save and output_path:
        cv2.imwrite(output_path, img)


def segmentation_by_color(input_path: str, lower_bound: tuple, upper_bound: tuple, output_path: str = '', save: bool = False, show: bool = False):
    """Segmenta una imagen basada en un rango de color en el espacio HSV."""

    # Cargar la imagen en color
    image = cv2.imread(input_path)

    # Verificar si la imagen se cargó correctamente
    if image is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo."
        )

    # Convertir a espacio de color HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Convertir las entradas de tuple a arrays numpy
    lower_bound = np.array(lower_bound, dtype=np.uint8)
    upper_bound = np.array(upper_bound, dtype=np.uint8)

    # Crear máscara para segmentar el color dentro del rango especificado
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Aplicar máscara sobre la imagen original
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    # Mostrar la imagen original y la segmentada
    fig, ax = plt.subplots(2, 1, figsize=(6, 10))

    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Imagen Original")
    ax[0].axis("off")

    ax[1].imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Imagen Segmentada por Color")
    ax[1].axis("off")

    if show:
        plt.show()

    # Guardar la imagen segmentada si se especifica
    if save and output_path:
        cv2.imwrite(output_path, segmented_image)


def segmentation_by_color_range(input_path: str, lower_bounds: list, upper_bounds: list, output_path: str = '', save: bool = False, show: bool = False):

    # Cargar la imagen en color
    image = cv2.imread(input_path)

    # Verificar si la imagen se cargó correctamente
    if image is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo."
        )

    # Convertir la imagen de BGR a HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Verificar que lower_bounds y upper_bounds tengan la misma cantidad de rangos
    if len(lower_bounds) != len(upper_bounds):
        raise ValueError(
            "Error: lower_bounds y upper_bounds deben tener la misma cantidad de elementos.")

    # Crear una máscara vacía
    mask = np.zeros_like(hsv[:, :, 0], dtype=np.uint8)

    # Aplicar segmentación para cada par de rangos de color
    for lower, upper in zip(lower_bounds, upper_bounds):
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        # Combinar las máscaras de los rangos
        mask |= cv2.inRange(hsv, lower, upper)

    # Aplicar la máscara a la imagen original
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    # Mostrar la imagen original y la segmentada
    fig, ax = plt.subplots(2, 1, figsize=(6, 10))

    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Imagen Original")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Máscara de Segmentación")
    ax[1].axis("off")

    if show:
        plt.show()

    # Guardar la imagen segmentada si se especifica
    if save and output_path:
        cv2.imwrite(output_path, segmented_image)

    # return segmented_image


def apply_binary_mask(input_path: str, minValue: int = 100, maxValue=255,  output_path: str = '', save: bool = False, show: bool = False):

    # Cargar la imagen en escala de grises
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(
            f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Crear una máscara binaria (por ejemplo, seleccionar píxeles que sean mayores que 100)
    _, mask = cv2.threshold(img, minValue, maxValue, cv2.THRESH_BINARY)

    # Aplicar la máscara a la imagen original usando bitwise_and
    result = cv2.bitwise_and(img, img, mask=mask)

    # Mostrar la imagen original, la máscara y el resultado
    plt.title('Segmentación de Color')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')  # Ocultar los ejes
    if show:
        plt.show()


# Guardar la imagen segmentada si se especifica
    if save and output_path:
        cv2.imwrite(output_path, mask)

    # return result
