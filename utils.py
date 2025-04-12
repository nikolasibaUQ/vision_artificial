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



#no sirve para maderas deja todo en rojo por la forma de las imagenes 
def detectar_lineas_hough(routPath, umbral_canny_bajo=50, umbral_canny_alto=150, aperture_size=7, umbral_hough=150, mostrar=False):
    """
    Aplica la Transformada de Hough para detectar líneas en una imagen.

    Parámetros:
        ruta_imagen (str): Ruta a la imagen a procesar.
        umbral_canny_bajo (int): Umbral bajo para Canny.
        umbral_canny_alto (int): Umbral alto para Canny.
        aperture_size (int): Tamaño del kernel Sobel (3, 5 o 7).
        umbral_hough (int): Umbral mínimo para detectar líneas con Hough.
        mostrar (bool): Si True, muestra la imagen con líneas.

    Retorna:
        image_result (np.ndarray): Imagen con líneas detectadas dibujadas.
        num_lineas (int): Número de líneas detectadas.
    """

    image = mt.getImage(routPath)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {routPath}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detección de bordes
    edges = cv2.Canny(gray, umbral_canny_bajo, umbral_canny_alto, apertureSize=aperture_size)

    # Transformada de Hough
    lines = cv2.HoughLines(edges, 1, np.pi / 20, umbral_hough)

    # Dibujar líneas
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            x1 = int(rho * np.cos(theta) + 1000 * (-np.sin(theta)))
            y1 = int(rho * np.sin(theta) + 1000 * (np.cos(theta)))
            x2 = int(rho * np.cos(theta) - 1000 * (-np.sin(theta)))
            y2 = int(rho * np.sin(theta) - 1000 * (np.cos(theta)))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Mostrar
    if mostrar:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Líneas Detectadas con la Transformada de Hough")
        plt.axis('off')
        plt.show()

    return image, len(lines) if lines is not None else 0

def descriptor_hog(img, show=False, resumen=True):
    img = mt.getImage(img)
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(img)

    if resumen:
        return {
            'HOG_mean': float(np.mean(hog_features)),
            'HOG_std': float(np.std(hog_features))
        }

    if show:
        plt.imshow(hog_features.reshape(-1, 1), cmap='gray')
        plt.title("Descriptor HOG")
        plt.axis('off')
        plt.show()

    return hog_features.flatten()



def kaze(img, show=False, resumen=True):
    img = mt.getImage(img)
    kaze = cv2.KAZE_create()
    keypoints, descriptors = kaze.detectAndCompute(img, None)

    if show:
        img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
        plt.imshow(img_keypoints, cmap=None)
        plt.title("KAZE Keypoints")
        plt.show()

    if resumen:
        return {
            'KAZE_count': len(keypoints),
            'KAZE_mean': float(descriptors.mean()) if descriptors is not None else 0,
            'KAZE_std': float(descriptors.std()) if descriptors is not None else 0
        }

    return descriptors  # matriz completa solo si se requiere


def orb_descriptor(path_img, show=False, resumen=True):
    """
    Calcula descriptores ORB y devuelve resumen estadístico.

    Parámetros:
        path_img (str): Ruta a la imagen.
        show (bool): Si True, muestra los puntos clave detectados.
        resumen (bool): Si True, retorna resumen estadístico (count, mean, std).

    Retorna:
        dict o ndarray: estadísticas (por defecto) o vector de descriptores.
    """
    img = mt.getImage(path_img)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {path_img}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Crear el detector ORB
    orb = cv2.ORB_create(nfeatures=500)

    # Detectar puntos clave y calcular descriptores
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if show:
        img_kp = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
        plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        plt.title("Puntos clave ORB")
        plt.axis("off")
        plt.show()

    if resumen:
        return {
            'ORB_count': len(keypoints),
            'ORB_mean': float(descriptors.mean()) if descriptors is not None else 0,
            'ORB_std': float(descriptors.std()) if descriptors is not None else 0
        }

    return descriptors


def akaze_descriptor(path_img, show=False, resumen=True):
    """
    Calcula descriptores AKAZE de una imagen.

    Retorna un resumen (por defecto) con count, media y desviación estándar.
    """
    img = mt.getImage(path_img)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {path_img}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    akaze = cv2.AKAZE_create()

    keypoints, descriptors = akaze.detectAndCompute(gray, None)

    if show:
        img_kp = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
        plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        plt.title("Puntos clave AKAZE")
        plt.axis("off")
        plt.show()

    if resumen:
        return {
            'AKAZE_count': len(keypoints),
            'AKAZE_mean': float(descriptors.mean()) if descriptors is not None else 0,
            'AKAZE_std': float(descriptors.std()) if descriptors is not None else 0
        }

    return descriptors


def log_features(path_img, resumen=True, show=False):
    """
    Aplica el filtro Laplaciano de Gauss para extraer detalles finos.

    Retorna un resumen estadístico del mapa de bordes.
    """
    img = mt.getImage(path_img)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {path_img}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    if show:
        plt.imshow(laplacian, cmap='gray')
        plt.title("Laplaciano de Gauss (LoG)")
        plt.axis('off')
        plt.show()

    if resumen:
        return {
            'LoG_mean': float(np.mean(laplacian)),
            'LoG_std': float(np.std(laplacian))
        }

    return laplacian
