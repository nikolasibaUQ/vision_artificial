
import cv2
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import metodos as mt
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte


default_outputs = 'parcial_2/output/'


def medium(img: str):
    """Calcula la textura de un objeto en una imagen y devuelve el valor medio de la textura.
    Args:
        img (str): Ruta de la imagen.
    Returns:
        float: Valor medio de la textura.
    """

    image = mt.getImage(img)

    return np.mean(image)


def variance(img: str):
    """Calcula la textura de un objeto en una imagen y devuelve la varianza de la textura.
    Args:
        img (str): Ruta de la imagen.
    Returns:
        float: Varianza de la textura.
    """
    # Cargar la imagen

    imgage = mt.getImage(img)

    return np.var(imgage)


def desviation(img: str):
    """Calcula la textura de un objeto en una imagen y devuelve la desviación estándar de la textura.
    Args:
        img (str): Ruta de la imagen.
    Returns:
        float: Desviación estándar de la textura.
    """
    # Obtener la imagen en escala de grises
    imgage = mt.getImage(img)

    return np.std(imgage)


def entropy(img: str):
    """Calcula la textura de un objeto en una imagen y devuelve la entropía de la textura.
    Args:
        img (str): Ruta de la imagen.
    Returns:
        float: Entropía de la textura.
    """
    # Cargar la imagen

    image_float = np.float32(mt.getImage(img)) + 1e-5
    entropy = -np.sum(image_float * np.log(image_float))

    return entropy


def get_images_bytes(img):
    """Convierte una imagen a bytes.
    Args:
        img (str): Ruta de la imagen.
    Returns:
        bytes: Imagen convertida a bytes.
    """
    image = mt.getImage(img)
    return img_as_ubyte(image)


def get_glcm(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
    """Calcula la matriz de co-ocurrencia de niveles de gris (GLCM) de una imagen."""
    image = mt.getImage(img)

    # Asegurarse de que sea una imagen en escala de grises
    if len(image.shape) == 3:
        import cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    from skimage.util import img_as_ubyte
    image = img_as_ubyte(image)  # Asegurar tipo adecuado para graycomatrix

    glcm = graycomatrix(image, distances=distances, angles=angles,
                        levels=levels, symmetric=symmetric, normed=normed)

    return glcm


def contrast(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
    """Calcula el contraste de la GLCM de una imagen.
    Args:
        img (str): Ruta de la imagen.
        distances (list): Distancias para calcular la GLCM.
        angles (list): Ángulos para calcular la GLCM.
        levels (int): Número de niveles de gris.
        symmetric (bool): Si la GLCM es simétrica.
        normed (bool): Si la GLCM está normalizada.
    Returns:
        numpy.ndarray: Contraste de la GLCM.
    """
    glcm = get_glcm(img, distances, angles, levels, symmetric, normed)

    contrast = graycoprops(glcm, 'contrast')

    return contrast[0][0]


def homogeneity(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
    """Calcula la homogeneidad de la GLCM de una imagen.
    Args:
        img (str): Ruta de la imagen.
        distances (list): Distancias para calcular la GLCM.
        angles (list): Ángulos para calcular la GLCM.
        levels (int): Número de niveles de gris.
        symmetric (bool): Si la GLCM es simétrica.
        normed (bool): Si la GLCM está normalizada.
    Returns:
        numpy.ndarray: Homogeneidad de la GLCM.
    """
    glcm = get_glcm(img, distances, angles, levels, symmetric, normed)

    homogeneity = graycoprops(glcm, 'homogeneity')

    return homogeneity[0][0]


def desimilarity(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
    """Calcula la disimilitud de la GLCM de una imagen.
    Args:
        img (str): Ruta de la imagen.
        distances (list): Distancias para calcular la GLCM.
        angles (list): Ángulos para calcular la GLCM.
        levels (int): Número de niveles de gris.
        symmetric (bool): Si la GLCM es simétrica.
        normed (bool): Si la GLCM está normalizada.
    Returns:
        numpy.ndarray: Disimilitud de la GLCM.
    """
    glcm = get_glcm(img, distances, angles, levels, symmetric, normed)

    dissimilarity = graycoprops(glcm, 'dissimilarity')

    return dissimilarity[0][0]


def energy(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
    """Calcula la energía de la GLCM de una imagen.
    Args:
        img (str): Ruta de la imagen.
        distances (list): Distancias para calcular la GLCM.
        angles (list): Ángulos para calcular la GLCM.
        levels (int): Número de niveles de gris.
        symmetric (bool): Si la GLCM es simétrica.
        normed (bool): Si la GLCM está normalizada.
    Returns:
        numpy.ndarray: Energía de la GLCM.
    """
    glcm = get_glcm(img, distances, angles, levels, symmetric, normed)

    energy = graycoprops(glcm, 'energy')

    return energy[0][0]


def correlation(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
    """Calcula la correlación de la GLCM de una imagen.
    Args:
        img (str): Ruta de la imagen.
        distances (list): Distancias para calcular la GLCM.
        angles (list): Ángulos para calcular la GLCM.
        levels (int): Número de niveles de gris.
        symmetric (bool): Si la GLCM es simétrica.
        normed (bool): Si la GLCM está normalizada.
    Returns:
        numpy.ndarray: Correlación de la GLCM.
    """
    glcm = get_glcm(img, distances, angles, levels, symmetric, normed)

    correlation = graycoprops(glcm, 'correlation')

    return correlation[0][0]


def mean_glcm(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
    """Calcula la media de la GLCM de una imagen.
    Args:
        img (str): Ruta de la imagen.
        distances (list): Distancias para calcular la GLCM.
        angles (list): Ángulos para calcular la GLCM.
        levels (int): Número de niveles de gris.
        symmetric (bool): Si la GLCM es simétrica.
        normed (bool): Si la GLCM está normalizada.
    Returns:
        numpy.ndarray: Media de la GLCM.
    """
    glcm = get_glcm(img, distances, angles, levels, symmetric, normed)

    mean = np.mean(glcm)

    return mean


def desviation_glcm(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
    """Calcula la desviación estándar de la GLCM de una imagen.
    Args:
        img (str): Ruta de la imagen.
        distances (list): Distancias para calcular la GLCM.
        angles (list): Ángulos para calcular la GLCM.
        levels (int): Número de niveles de gris.
        symmetric (bool): Si la GLCM es simétrica.
        normed (bool): Si la GLCM está normalizada.
    Returns:
        numpy.ndarray: Desviación estándar de la GLCM.
    """
    glcm = get_glcm(img, distances, angles, levels, symmetric, normed)

    deviation = np.std(glcm)

    return deviation


def calculate_entropy_glcm(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
    """Calcula la entropía de la GLCM de una imagen.
    Args:
        img (str): Ruta de la imagen.
        distances (list): Distancias para calcular la GLCM.
        angles (list): Ángulos para calcular la GLCM.
        levels (int): Número de niveles de gris.
        symmetric (bool): Si la GLCM es simétrica.
        normed (bool): Si la GLCM está normalizada.
    Returns:
        numpy.ndarray: Entropía de la GLCM.
    """
    glcm = get_glcm(img, distances, angles, levels, symmetric, normed)

    glcm_flat = glcm.flatten()
    glcm_flat = glcm_flat[glcm_flat > 0]  # Eliminar ceros para evitar log(0)
    entropy_glcm = -np.sum(glcm_flat * np.log(glcm_flat))

    return entropy_glcm


def get_glcm_features(image_path):
    try:
        img = imread(image_path)

        # Si la imagen ya es gris, no aplicar rgb2gray
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = rgb2gray(img)
        else:
            gray = img / 255.0  # Normalizar si ya es gris

        gray = (gray * 255).astype(np.uint8)

        glcm = graycomatrix(gray, distances=[1], angles=[
                            0], levels=256, symmetric=True, normed=True)

        return {
            'glcm_contrast': graycoprops(glcm, 'contrast')[0, 0],
            'glcm_homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
            'glcm_energy': graycoprops(glcm, 'energy')[0, 0],
            'glcm_correlation': graycoprops(glcm, 'correlation')[0, 0],
            'glcm_asm': graycoprops(glcm, 'ASM')[0, 0]
        }

    except Exception as e:
        print(f"❌ Error procesando GLCM para {image_path}: {e}")
        return {
            'glcm_contrast': np.nan,
            'glcm_homogeneity': np.nan,
            'glcm_energy': np.nan,
            'glcm_correlation': np.nan,
            'glcm_asm': np.nan
        }


def get_hog_features(image_path):
    try:
        img = imread(image_path)

        # Verificar si es RGB o ya está en escala de grises
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = rgb2gray(img)
        else:
            gray = img / 255.0  # Normalizar si ya es gris

        gray = (gray * 255).astype(np.uint8)

        features, _ = hog(
            gray,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            orientations=9,
            block_norm='L2-Hys',
            visualize=True
        )

        return {
            'hog_mean': np.mean(features),
            'hog_std': np.std(features),
            'hog_max': np.max(features),
            'hog_min': np.min(features)
        }

    except Exception as e:
        print(f"❌ Error procesando HOG para {image_path}: {e}")
        return {
            'hog_mean': np.nan,
            'hog_std': np.nan,
            'hog_max': np.nan,
            'hog_min': np.nan
        }


def get_laplacian_gauss_features(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {image_path}")

        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

        return {
            'laplacian_mean': np.mean(laplacian),
            'laplacian_std': np.std(laplacian),
            'laplacian_max': np.max(laplacian),
            'laplacian_min': np.min(laplacian)
        }

    except Exception as e:
        print(f"❌ Error procesando Laplaciano+Gauss para {image_path}: {e}")
        return {
            'laplacian_mean': np.nan,
            'laplacian_std': np.nan,
            'laplacian_max': np.nan,
            'laplacian_min': np.nan
        }
