import metodos as mt
import os
import pandas as pd
import parcial_1 as p1


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


def create_dataframe_recursively(base_path, etiqueta):
    data = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)
                # Genera la ruta relativa como en tu requerimiento
                relative_path = os.path.relpath(
                    full_path, start='proyecto_final')
                data.append({'filename': relative_path.replace(
                    '\\', '/'), 'etiqueta': etiqueta})
    return pd.DataFrame(data)


# Crear los dataframes para bueno y malo
df_bueno = create_dataframe_recursively('proyecto_final/outputs/good', 'bueno')
df_malo = create_dataframe_recursively('proyecto_final/outputs/sicks', 'malo')

# Concatenar y guardar
df = pd.concat([df_bueno, df_malo], ignore_index=True)
df.to_csv('proyecto_final/outputs/etiquetas.csv', index=False)

print("✅ Archivo 'etiquetas.csv' generado correctamente con imágenes de subcarpetas.")
