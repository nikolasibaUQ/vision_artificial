import cv2
from matplotlib import pyplot as plt
import metodos as mt
import os

default_inputRoute = 'parcial_1/resources/'
default_outputRoute = 'parcial_1/output/'

# def listar_imagenes_con_ruta(source_directory, output_base):
#     image_paths = []
    
#     for root_directory, _, archivos in os.walk(source_directory):  # Recorre subcarpetas
#         for archivo in archivos:
#             complete_file_path = os.path.join(root_directory, archivo)
            
#             if os.path.isfile(complete_file_path):
#                 # Obtener la subcarpeta relativa respecto a ruta_base
#                 relative_subdirectory = os.path.relpath(root_directory, source_directory)  # Ej: 'jackets', 'shoes', 'tshirts'

#                 # Construir la ruta de salida con la misma estructura
#                 output_subfolder_path = os.path.join(output_base, relative_subdirectory)
#                 os.makedirs(output_subfolder_path, exist_ok=True)  # Crea la subcarpeta si no existe

#                 # Construir la ruta final para la imagen convertida
#                 final_image_path = os.path.join(output_subfolder_path, archivo)

#                 # Llamar a la función para convertir la imagen
#                 mt.convert_gray_scale(complete_file_path, final_image_path, True)

#                 image_paths.append(final_image_path)

#     return image_paths



# contourn_img = mt.segmentation_contornos(input_path='parcial_1/resources/jackets/image_1.png', output_path='parcial_1/output/contornos.jpg',save= True)

# imagen_segmentada = mt.segmentation_kmeans(default_inputRoute + 'jackets/image_1.png' , default_outputRoute +'ejemplo.png'  , save=True, k=3)

# mt.segmentation_watershed('image.png' , default_outputRoute +'ejemplo.png' , save=True)
# mt.seed_points('image.png' , default_outputRoute +'ejemplo.png' , save=True)

# image = cv2.imread('image.png', 0)


# mt.segmentation_region_growing('image.png', 'output.png', save=True)




# mt.segmentation_watershed_contours('image.png', 'output.png', save=True)


# mt.segmentation_by_color('manzanas.png', lower_bound=(30, 50, 50), upper_bound=(80, 255, 255), output_path='output.png', save=True)
# mt.segmentation_by_color_range(
#     'manzanas.png',
#     lower_bounds=[[0, 120, 70], [170, 120, 70]],  # Rojo (dos rangos en HSV)
#     upper_bounds=[[10, 255, 255], [180, 255, 255]]
# )


mt.apply_binary_mask('manzanas.png', minValue=100, save=True, output_path='output.png')
