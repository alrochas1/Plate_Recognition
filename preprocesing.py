import cv2
import matplotlib.pyplot as plt

import aux



def show_histograms(before, after):
    
    original = cv2.calcHist([before], [0], None, [256], [0, 256])
    new = cv2.calcHist([after], [0], None, [256], [0, 256])
    
    plt.figure(figsize=(12, 6))
    
    # Imagen original y su histograma
    plt.subplot(2, 2, 1)
    plt.plot(original, color='black')
    plt.title("Histograma de Imagen Original")
    plt.xlabel("Intensidad de píxeles")
    plt.ylabel("Frecuencia")
    
    # Imagen con CLAHE y su histograma
    plt.subplot(2, 2, 2)
    plt.plot(new, color='black')
    plt.title("Histograma después")
    plt.xlabel("Intensidad de píxeles")
    plt.ylabel("Frecuencia")
    
    plt.tight_layout()
    plt.show()




def preprocess_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(20, 20))
    enhanced = clahe.apply(blurred)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)


    # Mostrar los resultados
    titles = ['Grises', 'Suavizada', 'Morfed']
    images = [gray, blurred, morphed]
    
    aux.plot_images(images, titles)

    # Mostrar histogramas
    plt.figure(figsize=(10, 5))
    plt.hist(gray.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7, label='Sin Ecualizar')
    plt.title('Histograma (Sin Ecualizar)')
    plt.xlabel('Intensidad de pixel')
    plt.ylabel('Frecuencia')
    plt.legend()

    plt.tight_layout()
    
    return gray


