import cv2
import numpy as np
import matplotlib.pyplot as plt

import aux

target_size=(516*2, 114*2)


def preprocess_image(image, show=True):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    # cv2.imwrite("./images/gray.jpg", gray)

    hist = cv2.calcHist([normalized], [0], None, [256], [0, 256])
    # peak = np.argmax(hist)
    average = np.sum(hist[:, 0] * np.arange(hist.shape[0])) / np.sum(hist)
    # print(average)

    # Aplicar solo si el histograma esta desplazado hacia la izquierda
    if average < 65:
        norm_eq = cv2.convertScaleAbs(normalized, alpha=3, beta=20)
    else:
        norm_eq = normalized

    gaussian = cv2.GaussianBlur(normalized, (5, 5), 0)
    median = cv2.medianBlur(norm_eq, 3)


    # Mostrar los resultados
    if show:
        titles = ['Grises', 'Ecualizado', 'Mediana']
        images = [normalized, norm_eq, median]
        
        aux.plot_images(images, titles)


        # gray = norm_eq
        # Mostrar histogramas
        plt.figure(figsize=(10, 5))
        plt.hist(gray.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7, label='Sin Ecualizar')
        plt.title('Histograma (Sin Ecualizar)')
        plt.xlabel('Intensidad de pixel')
        plt.ylabel('Frecuencia')
        plt.legend()

        plt.tight_layout()
    
    return median


