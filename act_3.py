import cv2
import matplotlib.pyplot as plt
import numpy as np



outputRout = 'act3/resources/'
# Cargar la imagen
image = cv2.imread('resources/images2.jpeg')

# Aplicar el filtro de desenfoque
blurred_image = cv2.blur(image, (50, 50))

# Mostrar la imagen filtrada con algún IDE de manera local
# cv2.imshow('Desenfoque', blurred_image)
# Esperar hasta que se cierre la ventana
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.title('Desenfoque')
plt.imshow(blurred_image)
plt.axis('off')  # Ocultar los ejes
plt.show()

# Guardar la imagen filtrada
cv2.imwrite( outputRout +'desenfoque.jpg' , blurred_image)



# Aplicar el filtro de desenfoque gaussiano
gaussian_blurred_image = cv2.GaussianBlur(image, (51, 51), 0)

# Mostrar la imagen filtrada
cv2.imshow('Desenfoque Gaussiano', gaussian_blurred_image)
# Esperar hasta que se cierre la ventana
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

plt.title('Desenfoque Gaussiano')
plt.imshow(gaussian_blurred_image)
plt.axis('off')  # Ocultar los ejes
plt.show()

# # Guardar la imagen filtrada
cv2.imwrite( outputRout+'imagen_gaussian_blur.jpg', gaussian_blurred_image)


# Cargar la imagen
# Crear un kernel para nítidez
sharpen_kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])

# Aplicar el filtro de nítidez el segundo parametro de profundidad de la imagen
# de salida ddepth
sharpened_image = cv2.filter2D(image, 0, sharpen_kernel)

# Mostrar la imagen filtrada
# cv2.imshow('Nitidez', sharpened_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


plt.title('Nitidez')
plt.imshow(sharpened_image)
plt.axis('off')  # Ocultar los ejes
plt.show()

# # Guardar la imagen filtrada
cv2.imwrite(outputRout+'imagen_nitida.jpg', sharpened_image)

# Esperar hasta que se cierre la ventana


# Cargar la imagen en escala de grises
image = cv2.imread('resources/images2.jpeg',
                   cv2.IMREAD_GRAYSCALE)

# Aplicar el filtro de detección de bordes (Canny)
edges = cv2.Canny(image, 100, 200)

# Mostrar la imagen filtrada
# cv2.imshow('Detección de Bordes (Canny)', edges)
# Esperar hasta que se cierre la ventana
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.title('Detección de Bordes (Canny)')
plt.imshow(edges)
plt.axis('off')  # Ocultar los ejes
plt.show()

# Guardar la imagen filtrada
cv2.imwrite(outputRout+'imagen_bordes.jpg', edges)




# Crear un kernel para el filtro de relieve
emboss_kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [0,  1, 2]])

# Aplicar el filtro de relieve
embossed_image = cv2.filter2D(image, -1, emboss_kernel)

# Mostrar la imagen filtrada
# cv2.imshow('Relieve', embossed_image)
# Esperar hasta que se cierre la ventana
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Guardar la imagen filtrada
cv2.imwrite(outputRout+'imagen_relieve.jpg', embossed_image)

plt.title('Relieve')
plt.imshow(embossed_image)
plt.axis('off')  # Ocultar los ejes
plt.show()


# # Cargar la imagen
image = cv2.imread('resources/images2.jpeg')

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Mostrar la imagen en escala de grises
# cv2.imshow('Escala de Grises', gray_image)
# Esperar hasta que se cierre la ventana
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Guardar la imagen filtrada
cv2.imwrite(outputRout+'imagen_gris.jpg', gray_image)

plt.title('Escala de Grises')
plt.imshow(gray_image)
plt.axis('off')  # Ocultar los ejes
plt.show()


# # Cargar la imagen en escala de grises
image = cv2.imread('resources/images2.jpeg',
                   cv2.IMREAD_GRAYSCALE)

# Umbralizar la imagen para obtener una imagen binaria
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Crear un kernel para la dilatación
kernel = np.ones((5, 5), np.uint8)

# Aplicar el filtro de dilatación
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

# Mostrar la imagen filtrada
# cv2.imshow('Dilatación', dilated_image)
# Esperar hasta que se cierre la ventana
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Guardar la imagen filtrada
cv2.imwrite(outputRout+'imagen_dilatada.jpg', dilated_image)

plt.title('Dilatación')
plt.imshow(dilated_image)
plt.axis('off')  # Ocultar los ejes
plt.show()
