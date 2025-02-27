# Codficación python
import cv2
import matplotlib.pyplot as plt

# Leer la imagen (en formato RGB)
img = cv2.imread('resources/images.jpeg')

# Convertir la imagen de BGR a RGB (OpenCV usa BGR por defecto)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Mostrar la imagen utilizando matplotlib (para trabajar con RGB)
plt.imshow(img_rgb)
plt.axis('off')  # Ocultar los ejes
plt.show()

# Guardar la imagen en un nuevo archivo
cv2.imwrite('images/imagen_guardada.jpg', img)




# Leer la imagen en escala de grises
img_gray = cv2.imread('resources/images2.jpeg', cv2.IMREAD_GRAYSCALE)


# Leer la imagen en blanco y negro (binaria) usando un umbral
_, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

# Leer la imagen original en RGB (se asume que la imagen es RGB)
img_rgb = cv2.imread('resources/images2.jpeg')

# Mostrar las imágenes utilizando matplotlib
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Escala de grises
axes[0].imshow(img_gray, cmap='gray')
axes[0].set_title("Escala de Grises")
axes[0].axis('off')

# Blanco y negro (binaria)
axes[1].imshow(img_binary, cmap='gray')
axes[1].set_title("Blanco y Negro")
axes[1].axis('off')

# Imagen RGB
axes[2].imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
axes[2].set_title("RGB")
axes[2].axis('off')

plt.show()


# Codificación en python
# Obtener la forma de la imagen
print("Forma de la imagen original:", img_rgb.shape)
print("Forma de la imagen en escala de grises:", img_gray.shape)
print("Forma de la imagen binaria:", img_binary.shape)


# Codificación python
# Guardar imágenes en diferentes formatos
cv2.imwrite('images/imagen_grises.jpg', img_gray)   # Imagen en escala de grises
cv2.imwrite('images/imagen_binaria.jpg', img_binary)  # Imagen binaria
cv2.imwrite('images/imagen_rgb.jpg', img_rgb)  # Imagen RGB

