import metodos as mt
import os
import pandas as pd
import parcial_1 as p1
import texturas as t
import matematic_metodes as mt

# obtener las imagenes de la carpeta
sicks = 'proyecto_final/resources/sick/'
good = 'proyecto_final/resources/good/'
default_outputs_sicks = 'proyecto_final/outputs/sicks/'
default_outputs_good = 'proyecto_final/outputs/good/'

# Obtener las imágenes de la carpeta resources
# images = os.listdir(sicks)
# for i in range(len(images)):

#     # procesar imagenes con el parcial 1
#     p1.procesar_imagen(
#         sicks + images[i], default_outputs_sicks + images[i].replace('.png', ''))

# images = os.listdir(good)
# for i in range(len(images)):

#     # procesar imagenes con el parcial 1
#     p1.procesar_imagen(
#         good + images[i], default_outputs_good + images[i].replace('.png', ''))


df = pd.read_csv('proyecto_final/outputs/etiquetas.csv')

# Extraer características
features = []
for _, row in df.iterrows():
    path = os.path.join('proyecto_final', row['filename'])
    feature_row = {
        'filename': row['filename'],
        'etiqueta': row['etiqueta'],
        'media': mt.get_media_image(path),
        'moda': mt.get_mode_image(path),
        'desviacion': mt.get_desviation_image(path)
    }
    feature_row.update(t.get_glcm_features(path))
    feature_row.update(t.get_hog_features(path))
    feature_row.update(t.get_laplacian_gauss_features(path))
    features.append(feature_row)

# Crear DataFrame final
df_features = pd.DataFrame(features)

# Guardar CSV
df_features.to_csv(
    'proyecto_final/outputs/dataset_caracteristicas.csv', index=False)
