import os
import texturas as t
import pandas as pd
import parcial_1 as p1

default_inputs = 'parcial_2/resources/'
default_outputs = 'parcial_2/output/'


# Obtener las imágenes de la carpeta resources
images = os.listdir(default_inputs)
for i in range(len(images)):
    
# procesar imagenes con el parcial 1
    p1.procesar_imagen(default_inputs + images[i] , default_outputs + images[i])


ruta_output = 'parcial_2/output/'

# Recorre todos los elementos en la carpeta 'output'
for nombre_carpeta in os.listdir(ruta_output):
    ruta_completa = os.path.join(ruta_output, nombre_carpeta)

    # Verifica que sea una carpeta y termine en .png
    if os.path.isdir(ruta_completa) and nombre_carpeta.endswith('.png'):
        nuevo_nombre = nombre_carpeta.replace('.png', '')
        nueva_ruta = os.path.join(ruta_output, nuevo_nombre)

        # Renombrar la carpeta
        os.rename(ruta_completa, nueva_ruta)
        print(f"✅ Renombrada: {nombre_carpeta} → {nuevo_nombre}")

default_outputs = 'parcial_2/output/'

# Recorremos cada carpeta dentro de 'output'
for nombre_carpeta in os.listdir(default_outputs):
    ruta_carpeta = os.path.join(default_outputs, nombre_carpeta)

    if not os.path.isdir(ruta_carpeta):
        continue  # saltar si no es carpeta

    datos = []

    # Recorremos cada imagen dentro de esa carpeta
    for nombre_imagen in os.listdir(ruta_carpeta):
        ruta_imagen = os.path.join(ruta_carpeta, nombre_imagen)

        if not nombre_imagen.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # ignorar archivos no imagen

        try:
            # Aplicar análisis de texturas a la imagen
            media = t.medium(ruta_imagen)
            varianza = t.variance(ruta_imagen)
            desviacion = t.desviation(ruta_imagen)
            entropia = t.entropy(ruta_imagen)
            contraste = t.contrast(ruta_imagen)
            homogeneidad = t.homogeneity(ruta_imagen)
            disimilitud = t.desimilarity(ruta_imagen)
            energia = t.energy(ruta_imagen)
            correlacion = t.correlation(ruta_imagen)
            media_glcm = t.mean_glcm(ruta_imagen)
            desviacion_glcm = t.desviation_glcm(ruta_imagen)
            entropia_glcm = t.calculate_entropy_glcm(ruta_imagen)

            # Guardamos en la lista
            datos.append({
                'Imagen': nombre_imagen,
                'Varianza': varianza,
                'Media': media,
                'Desviación estándar': desviacion,
                'Entropía': entropia,
                'Contraste': contraste,
                'Homogeneidad': homogeneidad,
                'Disimilitud': disimilitud,
                'Energía': energia,
                'Correlación': correlacion,
                'Media GLCM': media_glcm,
                'Desviación GLCM': desviacion_glcm,
                'Entropía GLCM': entropia_glcm
            })
        except Exception as e:
            print(f"⚠️ Error al procesar {ruta_imagen}: {e}")

    # Crear DataFrame y guardar el Excel solo si hay datos
    if datos:
        df = pd.DataFrame(datos)
        output_excel = os.path.join(ruta_carpeta, 'resultados_textura.xlsx')
        df.to_excel(output_excel, index=False)
        print(f"✅ Excel guardado en: {output_excel}")
    else:
        print(f"⚠️ No se procesaron imágenes en: {ruta_carpeta}")