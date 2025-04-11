import os
import random
import cv2
import metodos as mt  # Importamos el archivo de métodos

# Directorios de entrada y salida
input_base = "parcial_2/resources/"
output_base = "parcial_2/output/"

os.makedirs(output_base, exist_ok=True)


def procesar_imagen(input_path, output_folder):
    """
    Aplica todos los métodos de procesamiento a una imagen y guarda los resultados en una subcarpeta.
    """
    os.makedirs(output_folder, exist_ok=True)

    image = cv2.imread(input_path)
    if image is None:
        print(f"Error al cargar {input_path}")
        return None

    original_path = os.path.join(output_folder, "original.png")
    cv2.imwrite(original_path, image)

    # Cambios de tonalidad
    mt.convert_RGB(input_path, os.path.join(
        output_folder, "RGB.png"), save=True)

    # Transformaciones
    gray_path = os.path.join(output_folder, "gray_scale.png")
    binary_path = os.path.join(output_folder, "binary.png")
    binary_resized_path = os.path.join(output_folder, "binary_resized.png")

    mt.convert_gray_scale(input_path, gray_path, save=True)
    mt.convert_binary_image(input_path, binary_path, save=True)
    mt.resize_image(binary_path, binary_resized_path,
                    save=True, width=600, height=600)

    # Filtros
    mt.blur_image(input_path, os.path.join(
        output_folder, "blur.png"), save=True, kernel_size=5)
    mt.gaussian_blur_image(input_path, os.path.join(
        output_folder, "gaussian_blur.png"), save=True, kernel_size=5)
    mt.sharpen_image(input_path, os.path.join(
        output_folder, "sharpen.png"), save=True)
    mt.detect_edges(input_path, os.path.join(
        output_folder, "edges.png"), save=True)
    mt.emboss_image(input_path, os.path.join(
        output_folder, "emboss.png"), save=True)

    # Operaciones Morfológicas
    mt.delete_noides_opened_image(input_path, os.path.join(
        output_folder, "morph_open.png"), save=True)
    mt.delete_noides_closed_image(input_path, os.path.join(
        output_folder, "morph_close.png"), save=True)
    mt.gradient_image(input_path, os.path.join(
        output_folder, "gradient.png"), save=True)
    mt.dilated_image(input_path, os.path.join(
        output_folder, "dilated.png"), save=True)

    # Ampliación y Reducción
    mt.resize_image(input_path, os.path.join(
        output_folder, "resize.png"), save=True, width=600, height=600)
    mt.rotate_image(input_path, os.path.join(
        output_folder, "rotate.png"), save=True, angle= random.randint(0, 280))

    # Segmentación
    mt.segmentation_umbral_image(input_path, os.path.join(
        output_folder, "segmentation_umbral.png"), save=True)
    mt.segmentation_kmeans(input_path, os.path.join(
        output_folder, "segmentation_kmeans.png"), save=True)
    mt.segmentation_watershed(input_path, os.path.join(
        output_folder, "segmentation_watershed.png"), save=True)
    mt.segmentation_region_growing(input_path, os.path.join(
        output_folder, "segmentation_region_growing.png"), save=True)
    mt.segmentation_watershed_contours(input_path, os.path.join(
        output_folder, "segmentation_watershed_contours.png"), save=True)
    mt.segmentation_by_color(input_path, (0, 40, 40), (20, 255, 255), os.path.join(
        output_folder, "segmentation_by_color.png"), save=True)

    # Segmentación por múltiples colores (Rojo, Verde y Azul)
    mt.segmentation_by_color_range(input_path,
                                   [(0, 40, 40), (160, 40, 40), (35, 40, 40),
                                    (90, 50, 50)],  # Rojos, Verde, Azul
                                   [(20, 255, 255), (180, 255, 255),
                                    (85, 255, 255), (130, 255, 255)],
                                   os.path.join(
                                       output_folder, "segmentation_by_color_range.png"),
                                   save=True)

    return binary_resized_path


# Recorrer todas las carpetas dentro de la carpeta de recursos
for categoria in os.listdir(input_base):
    categoria_path = os.path.join(input_base, categoria)

    if os.path.isdir(categoria_path):
        print(f"Procesando categoría: {categoria}")

        imagenes = sorted([img for img in os.listdir(
            categoria_path) if img.lower().endswith((".png", ".jpg", ".jpeg"))])

        if len(imagenes) < 1:
            continue

        imagenes_procesadas = []

        for imagen_nombre in imagenes:
            imagen_path = os.path.join(categoria_path, imagen_nombre)

            nombre_sin_extension = os.path.splitext(imagen_nombre)[0]
            output_folder = os.path.join(
                output_base, categoria, nombre_sin_extension)

            binary_resized = procesar_imagen(imagen_path, output_folder)
            if binary_resized:
                imagenes_procesadas.append((binary_resized, output_folder))

        if len(imagenes_procesadas) > 1:
            for i in range(len(imagenes_procesadas)):
                bin1, folder1 = imagenes_procesadas[i]
                bin2, folder2 = imagenes_procesadas[(
                    i + 1) % len(imagenes_procesadas)]

                print(
                    f"Comparando {os.path.basename(folder1)} con {os.path.basename(folder2)}")

                output_folder_op = os.path.join(folder1, "comparaciones")
                os.makedirs(output_folder_op, exist_ok=True)

                mt.add_images(bin1, bin2, os.path.join(
                    output_folder_op, "add.png"), save=True)
                mt.subtract_images(bin1, bin2, os.path.join(
                    output_folder_op, "subtract.png"), save=True)
                mt.and_images(bin1, bin2, os.path.join(
                    output_folder_op, "and.png"), save=True)
                mt.or_images(bin1, bin2, os.path.join(
                    output_folder_op, "or.png"), save=True)

print("Procesamiento completado.")
