import cv2
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

import aux

target_size=(200, 50)


# Para recortar la matricula
def cut_plate(image):

    edges = cv2.Canny(image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos por área y forma
    possible_plates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)
        if 2 < aspect_ratio < 6 and area > 1000:  # Ajustar estos valores según tus datos
            possible_plates.append((x, y, w, h))
    
    # Seleccionar el contorno más grande que cumpla los criterios
    if not possible_plates:
        print("No se detecto nada a recortar (o no hay matricula o esta recortada)")
        return image
    else:
        x, y, w, h = max(possible_plates, key=lambda b: b[2] * b[3])  # Por área
        # Recortar la región de interés
        plate = image[y:y+h, x:x+w]
        return plate



def segment_characters(image):

    cropped_plate = cut_plate(image)
    normalized_plate = cv2.resize(cropped_plate, target_size, interpolation=cv2.INTER_AREA)

    # Aplicar Otsu
    _, binary = cv2.threshold(normalized_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)    # Hace falta invertirlo para la siguiente parte

    # Etiquetado de regiones conectadas
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Filtrar regiones
    min_area = 110; max_area = 550
    min_ratio = 0.05; max_ratio = 5
    char_images = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        # TODO: Filtrar tambien por otras propiedades (relacion alto y ancho)
        if min_area < area < max_area:
            ratio = stats[i, cv2.CC_STAT_WIDTH] / stats[i, cv2.CC_STAT_HEIGHT]
            # print(ratio)    # Facil --> 0.25 a 0.55, Medio --> 0.14 a 0.65 (los malos a 7 y 75)
            if min_ratio < ratio < max_ratio:
                char = binary[y:y+h, x:x+w]
                # Redimensionar para normalizar tamaño
                char = aux.normalize(char)
                char = (stats[i, cv2.CC_STAT_LEFT], cv2.bitwise_not(char))    # Lo invierto para que se vea mejor
                char_images.append(char)
        
    
    # Se ordena segun el eje X
    # char_images.sort(key=itemgetter(0))
    sorted = [char for _, char in char_images]

    # Mostrar los resultados
    titles = ['Original', 'Cropped Plate', 'Normalizada', 'Otsu']
    images = [image, cropped_plate, normalized_plate, binary]
    aux.plot_images(images, titles)
    

    # Mostrar caracteres segmentados
    plt.figure(figsize=(10, 2))
    for i, j in enumerate(sorted):
        plt.subplot(1, len(sorted), i + 1)
        plt.imshow(j, cmap='gray')
        plt.axis('off')
    plt.suptitle('Caracteres Segmentados')
    plt.tight_layout()
    
    return sorted

