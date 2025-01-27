import cv2
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

import aux

target_size=(258, 57)


def segment_characters(image, show=True):

    normalized_plate = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # # Aplicar Otsu
    _, binary = cv2.threshold(normalized_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)    # Hace falta invertirlo para la siguiente parte

    # Etiquetado de regiones conectadas
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Filtrar regiones
    min_area = 185; max_area = 850
    min_ratio = 0.05; max_ratio = 0.9   # No hay letras mas anchas que altas
    char_images = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
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
    char_images.sort(key=itemgetter(0))
    sorted = [char for _, char in char_images]

    if show:
        # Mostrar los resultados
        titles = ['Normalizada', 'Otsu']
        images = [normalized_plate, binary]
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

