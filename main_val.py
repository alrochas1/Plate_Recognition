import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from preprocesing import preprocess_image
from correction import correct_image
from segmentation import segment_characters


folder_path = "./plates/train"

# Para almacenar resultados
correct_plates = 0
total_plates = 0
incorrect_plates = 0  # Aqui falla el bajo nivel
partial_plates = 0    # Aqui falla el alto nivel
    
model = tf.keras.models.load_model('./neural_model/nn_model.keras')
classes = list(map(str, range(10))) + [chr(i) for i in range(ord('A'), ord('Z') + 1)]

files = os.listdir(folder_path)
images = [archivo for archivo in files if archivo.lower().endswith(('.png', '.jpg', '.jpeg'))]
n_images = len(images)


with open(f"results.txt", "w") as results_file:
    for i, file_name in enumerate(images):
        print(f"{i+1}/{n_images}")
        total_plates += 1
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"No se pudo cargar la imagen {file_name}. Verifica la ruta.")
            continue

        # Procesar la imagen
        processed_image = preprocess_image(image, False)
        corrected_image = correct_image(processed_image, False)
        characters = segment_characters(corrected_image, False)
        plt.close('all') # Evita que use toda la RAM

        # Reconocer caracteres
        plate_number = ""
        for char_img in characters:
            char_img = char_img / 255.0
            char_img = char_img.reshape(1, 28, 28, 1)   # Esto en teoria no hace falta

            prediction = model.predict(char_img, verbose=0)
            predicted_class = np.argmax(prediction)
            plate_number += classes[predicted_class]

        # Obtener el número de matrícula real del nombre del archivo
        real_number, _ = os.path.splitext(file_name)

        if real_number == plate_number:
            correct_plates += 1
            status = "OK"
        else:
            if len(plate_number) < len(real_number)-2:  # No se reconocen varios caracteres
                incorrect_plates += 1
                status = "FALLO SEGMENTACION"
            else:  # Reconocimiento parcial
                partial_plates += 1
                status = "FALLO NN"

        # Libera memoria
        results_file.write(f"Archivo: {file_name}, Reconocido: {plate_number}, Estado: {status}\n")
        del image, processed_image, corrected_image, characters
        tf.keras.backend.clear_session()


# Mostrar resultados
print(f"Total de matrículas procesadas: {total_plates}")
print(f"Matrículas correctas: {correct_plates}")
print(f"Porcentaje de acierto: {(correct_plates / total_plates) * 100:.2f}%")
print(f"Matrículas sin reconocimiento: {incorrect_plates} ({(incorrect_plates / total_plates) * 100:.2f}%)")
print(f"Matrículas con reconocimiento parcial: {partial_plates} ({(partial_plates / total_plates) * 100:.2f}%)")
print("\nDetalles por matrícula:")
# for result in results:
#     print(f"Archivo: {result[0]}, Reconocido: {result[1]}, Estado: {result[2]}")

# plt.show() # No se muestran los resultados porque serian cientos de ventanas