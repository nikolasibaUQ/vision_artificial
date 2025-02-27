import cv2
import matplotlib.pyplot as plt
import numpy as np


# Cargar una imagen en color
img = cv2.imread('resources/images3.jpeg')
img2 = cv2.imread('resources/images4.jpeg')

# Convertir a escala de grises
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Mostrar la imagen resultante
plt.imshow(gray_img)
plt.axis('off')  # Ocultar los ejes
# plt.show()



# Convertir a escala de grises
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Umbralizar la imagen para convertirla en blanco y negro (binaria)
_, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)

# Mostrar las imágenes utilizando matplotlib
fig, axes = plt.subplots(2, 1, figsize=(15, 5))

# Escala de grises
axes[0].imshow(gray_img, cmap='gray')
axes[0].set_title("Escala de Grises")
axes[0].axis('off')

# Blanco y negro (binaria)
axes[1].imshow(binary_img, cmap='gray')
axes[1].set_title("Blanco y Negro")
axes[1].axis('off')

# plt.show()


# Cambiar el tamaño de la imagen (Reducción y Amplificación)
resized_img = cv2.resize(img, (400, 400))  # Cambiar el tamaño a 400x400 píxeles

# Mostrar la imagen redimensionada
plt.imshow(resized_img)
plt.axis('off')  # Ocultar los ejes
plt.show()


# Obtener las dimensiones de la imagen
(h, w) = img.shape[:2]

# Establecer el centro de la imagen para rotarla
center = (w // 2, h // 2)

# Crear la matriz de rotación (por ejemplo, 45 grados)
M = cv2.getRotationMatrix2D(center, 45, 1.0)

# Rotar la imagen
rotated_img = cv2.warpAffine(img, M, (w, h))

# Mostrar la imagen rotada
plt.imshow(rotated_img)
plt.axis('off')  # Ocultar los ejes
plt.show()


#modificacion del codigo original, para solo asignar una variable
img1 = img


# Suma de imágenes
sum_img = cv2.add(img1, img2)

# Resta de imágenes
diff_img = cv2.subtract(img1, img2)

# Multiplicación de imágenes
mult_img = cv2.multiply(img1, img2)

# División de imágenes
div_img = cv2.divide(img1, img2)

# Mostrar las imágenes resultantes
fig, axes = plt.subplots(2,2 , figsize=(15, 5))

axes[0][0].imshow(sum_img, cmap='gray')
axes[0][0].set_title("Suma")
axes[0][0].axis('off')

axes[0][1].imshow(diff_img, cmap='gray')
axes[0][1].set_title("Resta")
axes[0][1].axis('off')

axes[1][0].imshow(mult_img, cmap='gray')
axes[1][0].set_title("Multiplicación")
axes[1][0].axis('off')

axes[1][1].imshow(div_img, cmap='gray')
axes[1][1].set_title("División")
axes[1][1].axis('off')

plt.show()



# Cargar dos imágenes binarizadas (deben ser del mismo tamaño)
img1 = cv2.imread('resources/images.jpeg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('resources/images4.jpeg', cv2.IMREAD_GRAYSCALE)

# Convertir a binario (umbralizar si es necesario)
_, img1_bin = cv2.threshold(img1, 128, 255, cv2.THRESH_BINARY)
_, img2_bin = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY)

# Operación AND
and_img = cv2.bitwise_and(img1_bin, img2_bin)

# Operación OR
or_img = cv2.bitwise_or(img1_bin, img2_bin)

# Operación NOT
not_img1 = cv2.bitwise_not(img1_bin)
not_img2 = cv2.bitwise_not(img2_bin)

# Mostrar las imágenes resultantes

# Mostrar las imágenes utilizando matplotlib
fig, axes = plt.subplots(2,2 , figsize=(15, 5))

axes[0][0].imshow(and_img, cmap='gray')
axes[0][0].set_title("AND")
axes[0][0].axis('off')

axes[0][1].imshow(or_img, cmap='gray')
axes[0][1].set_title("OR")
axes[0][1].axis('off')

axes[1][0].imshow(not_img1, cmap='gray')
axes[1][0].set_title("NOT IMAGEN UNO")
axes[1][0].axis('off')

axes[1][1].imshow(not_img2, cmap='gray')
axes[1][1].set_title("NOT IMAGEN DOS")
axes[1][1].axis('off')

plt.show()



# Cambiar tamaño usando diferentes métodos de interpolación
resized_bilinear = cv2.resize(img, (400, 400), interpolation=cv2.INTER_LINEAR)  # Bilineal
resized_nearest = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)  # Vecino más cercano

# Mostrar las imágenes utilizando matplotlib
fig, axes = plt.subplots(2, 1, figsize=(15, 5))

# Escala de grises
axes[0].imshow(resized_bilinear, cmap='gray')
axes[0].set_title("Interpolación Bilineal")
axes[0].axis('off')

# Blanco y negro (binaria)
axes[1].imshow(resized_nearest, cmap='gray')
axes[1].set_title("Interpolación Vecino más cercano")
axes[1].axis('off')

plt.show()