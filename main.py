import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from preprocesing import preprocess_image
from segmentation import segment_characters
from correction import correct_perspective


# Ruta de la imagen de prueba (de mas facil a menos)
folder_path = "./plates/test"
# file_name = "5126HVL.png"
file_name = '1033IR.png'
# file_name = 'AL193VP.jpg'  # Aqui los numeros si, las letras se juntan
# file_name = 'DANKE82.png'  # Esta por algun motivo la desordena
# file_name = 'EH577PH.jpg'  # Esta tambien
# file_name = 'EQ725QJ.jpg'   # Esta tambien
# file_name = 'SLF9995.png'

image_path = os.path.join(folder_path, file_name)
image = cv2.imread(image_path)
if image is None:
    raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")


processed_image = preprocess_image(image)
characters = segment_characters(processed_image)
# corrected = correct_perspective(characters)


######################################################################################

# Cargar el modelo guardado
model = tf.keras.models.load_model('./neural_model/nn_model.keras')
classes = list(map(str, range(10))) + [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Reconocer caracteres
plate_number = ""
for char_img in characters:
    char_img = char_img / 255.0
    char_img = char_img.reshape(1, 28, 28, 1)

    # Predecir la clase del carácter
    prediction = model.predict(char_img)
    predicted_class = np.argmax(prediction)
    plate_number += classes[predicted_class]


# Muestra resultados
real_number, _ = os.path.splitext(file_name)
print(f"Numero de matricula Real: {real_number}")
print(f"Número de matrícula Reconocido: {plate_number}")
if real_number == plate_number:
    print("Matricula OK")
else:
    print("Matricula INCORRECTA")

plt.show()


