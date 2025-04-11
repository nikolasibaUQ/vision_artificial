import cv2
import numpy as np
import matplotlib.pyplot as plt
import metodos as mt


def momentos_hu(path_img: str, show: bool = False):
    """Devuelve los momentos de Hu de una imagen procesada adecuadamente."""

    img = mt.getImage(path_img)

    # Asegúrate de que la imagen es de un canal (blanco y negro)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Opcional: aplica un umbral para convertirla en binaria

    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    if show:
        plt.imshow(binary_img, cmap='gray')
        plt.axis('off')
        plt.show()
    # Calcular los momentos
    moments = cv2.moments(binary_img)
    return cv2.HuMoments(moments)
