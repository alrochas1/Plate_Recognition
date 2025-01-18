import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from preprocesing import preprocess_image
from segmentation import segment_characters
from correction import correct_perspective


# Ruta de la imagen de prueba (de mas facil a menos)
image_path = './plates/test/5126HVL.png' 
# image_path = './plates/test/1033IR.png'
# image_path = './plates/test/AL193VP.jpg'  # Aqui los numeros si, las letras se juntan
# image_path = './plates/test/DANKE82.png'  # Esta por algun motivo la desordena
# image_path = './plates/test/EH577PH.jpg'  # Esta tambien
# image_path = './plates/test/EQ725QJ.jpg'   # Esta tambien
# image_path = './plates/test/SLF9995.png'

image = cv2.imread(image_path)
if image is None:
    raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")


processed_image = preprocess_image(image)
characters = segment_characters(processed_image)
# corrected = correct_perspective(characters)


# Cargar el modelo guardado y los datos
model = tf.keras.models.load_model('./neural_model/nn_model.keras')
classes = list(map(str, range(10))) + [chr(i) for i in range(ord('A'), ord('Z') + 1)]
# plt.figure()
# plt.imshow(characters[0], cmap='gray')
# plt.show()

# Reconocer caracteres
plate_number = ""
for char_img in characters:
    char_img = char_img / 255.0
    char_img = char_img.reshape(1, 28, 28, 1)

    # Predecir la clase del carácter
    prediction = model.predict(char_img)
    predicted_class = np.argmax(prediction)
    plate_number += classes[predicted_class]


print(f"Número de matrícula reconocido: {plate_number}")

plt.show()


