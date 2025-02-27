import cv2
import matplotlib.pyplot as plt
import numpy as np

import cv2
import os

def convertRGB(input_path: str, output_path: str = '', save: bool = False):
    

    # Cargar la imagen
    img = cv2.imread(input_path)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

    # Convertir de BGR a RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Guardar la imagen convertida
    if  output_path != '' and save:
        output_path = os.path.abspath(output_path)  # Asegurar salida absoluta
        cv2.imwrite(output_path, img_rgb)

    # Mostrar la imagen si se solicita
    
    cv2.imshow('Imagen RGB', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
def grayImage( input_path: str, output_path: str = '', save: bool = False):
    
    # Cargar la imagen
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    
    
    if  output_path != '' and save:
        output_path = os.path.abspath(output_path)
        cv2.imwrite(output_path, img)
    
    cv2.imshow('Imagen en escala de grises', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def binaryImage(input_path: str, output_path: str = '', save: bool = False, threshold: int = 1, max_value: int = 255):
    
    # Cargar la imagen
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        raise ValueError(f"Error: No se pudo cargar la imagen '{input_path}'. Verifica la ruta y el formato del archivo.")

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
    
    
def normalImage( input_path: str, ):
    
    input_path = os.path.abspath(input_path)
    img = cv2.imread(input_path)
    
    
    
    plt.imshow(cv2.cvtColor( img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    


    
    




    
    
 