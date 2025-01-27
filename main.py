import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from preprocesing import preprocess_image
from correction import correct_image
from segmentation import segment_characters


# Ruta de la imagen de prueba (de mas facil a menos)
folder_path = "./plates/test"
file_name = "5126HVL.png"   # OK
# file_name = '1033IR.png'    # OK
# file_name = 'AL193VP.jpg'  # Aqui los numeros si, las letras se juntan
# file_name = 'DANKE82.png'  # OK (falla la NN con la E)
# file_name = 'EH577PH.jpg' # No la recorta bien, se come la primera H
# file_name = 'EQ725QJ.jpg'
# file_name = '41JA34.png'
file_name = 'KYM3141.png'   # Analizar porque no segmenta esto


image_path = os.path.join(folder_path, file_name)
image = cv2.imread(image_path)
if image is None:
    raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")


processed_image = preprocess_image(image)
corrected_image = correct_image(processed_image)
characters = segment_characters(corrected_image)


######################################################################################

GUESS_NUMBERS = True    # Cambiar esto para que use la NN (quitar para hacer pruebas)
if GUESS_NUMBERS:
    import tensorflow as tf
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


