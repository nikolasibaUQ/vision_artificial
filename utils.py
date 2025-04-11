# archivo: momentos_hu.py
import cv2
import numpy as np

def momentos_hu(path_img):
    """Devuelve los 7 momentos de Hu de una imagen."""
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 107, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(binary, 100, 200)
    moments = cv2.moments(canny)
    hu = cv2.HuMoments(moments).flatten()
    return hu.tolist()


# archivo: hog_descriptor.py
import cv2
import numpy as np

def hog_descriptor(path_img):
    """Devuelve estadísticas (media, desviación) del descriptor HOG."""
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    hog = cv2.HOGDescriptor()
    features = hog.compute(img)
    return {
        'hog_media': float(np.mean(features)),
        'hog_desviacion': float(np.std(features))
    }


# archivo: orb_keypoints.py
import cv2

def orb_keypoints(path_img):
    """Devuelve la cantidad de keypoints detectados por ORB."""
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    keypoints, _ = orb.detectAndCompute(img, None)
    return len(keypoints)


# archivo: akaze_keypoints.py
import cv2

def akaze_keypoints(path_img):
    """Devuelve la cantidad de keypoints detectados por AKAZE."""
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    akaze = cv2.AKAZE_create()
    keypoints, _ = akaze.detectAndCompute(img, None)
    return len(keypoints)


# archivo: kaze_keypoints.py
import cv2

def kaze_keypoints(path_img):
    """Devuelve la cantidad de keypoints detectados por KAZE."""
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    kaze = cv2.KAZE_create()
    keypoints, _ = kaze.detectAndCompute(img, None)
    return len(keypoints)

default_img = 'parcial_2/resources/image.png'

# llamar todos los metodos 

print(momentos_hu(default_img))
print(hog_descriptor(default_img))
print(orb_keypoints(default_img))
print(akaze_keypoints(default_img))
print(kaze_keypoints(default_img))
# print(entropy(default_img))