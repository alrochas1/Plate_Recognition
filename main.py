import cv2
import matplotlib.pyplot as plt

from preprocesing import preprocess_image
from segmentation import segment_characters
from correction import correct_perspective


# Ruta de la imagen de prueba (de mas facil a menos)
# image_path = './plates/test/5126HVL.png' 
# image_path = './plates/test/1033IR.png'
image_path = './plates/test/AL193VP.jpg'  # Aqui los numeros si, las letras se juntan
# image_path = './plates/test/DANKE82.png'  # Esta por algun motivo la desordena
# image_path = './plates/test/EH577PH.jpg'  # Esta tambien
# image_path = './plates/test/EQ725QJ.jpg'   # Esta tambien

image = cv2.imread(image_path)
if image is None:
    raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")


processed_image = preprocess_image(image)
characters = segment_characters(processed_image)
# corrected = correct_perspective(characters)

plt.show()


