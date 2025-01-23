import cv2
import matplotlib.pyplot as plt

import aux


def preprocess_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    target_size=(516, 114)
    normalized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    gaussian = cv2.GaussianBlur(normalized, (5, 5), 0)
    median = cv2.medianBlur(normalized, 3)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(20, 20))
    enhanced = clahe.apply(median)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)


    # Mostrar los resultados
    titles = ['Grises', 'Gaussiana', 'Mediana']
    images = [normalized, gaussian, median]
    
    aux.plot_images(images, titles)

    # Mostrar histogramas
    plt.figure(figsize=(10, 5))
    plt.hist(gray.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7, label='Sin Ecualizar')
    plt.title('Histograma (Sin Ecualizar)')
    plt.xlabel('Intensidad de pixel')
    plt.ylabel('Frecuencia')
    plt.legend()

    plt.tight_layout()
    
    return median


