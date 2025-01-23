import cv2
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

import aux

target_size=(258, 57)


def detect_rotation_angle(image):

    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is None:
        print("No se detectaron líneas en la imagen.")
        return 0, edges
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)
    
    median_angle = np.median(angles)
    return median_angle, edges

def correct_rotation(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def correct_perspective(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Aproximar el contorno a un polígono
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) == 4:
        # Ordenar los puntos para la transformación de perspectiva
        rect = np.zeros((4, 2), dtype="float32")
        s = approx.sum(axis=2)
        rect[0] = approx[np.argmin(s)]
        rect[2] = approx[np.argmax(s)]
        
        diff = np.diff(approx, axis=2)
        rect[1] = approx[np.argmin(diff)]
        rect[3] = approx[np.argmax(diff)]
        
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    else:
        return image



# Para recortar la matricula
# def cut_plate(image):

#     edges = cv2.Canny(image, 50, 150)
#     edges_border = cv2.copyMakeBorder(edges, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
#     edges_border = cv2.copyMakeBorder(edges_border, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
#     contours, _ = cv2.findContours(edges_border, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Filtrar contornos por área y forma
#     possible_plates = []
#     possible_contours = []
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         aspect_ratio = w / float(h)   # Deberia ser ~4.5
#         # area = cv2.contourArea(cnt)   # ~20000 
#         area = w*h      # El maximo es 58824
#         if (3 < aspect_ratio < 6.5) and (20000 < area < 50000): 
#             possible_plates.append((x, y, w, h))
#             possible_contours.append(cnt)
#             # print(f"Area: {area}")
#             # print(f"Aspect Ratio: {aspect_ratio}")
#             # print(f"x = {x}, y = {y}")
#             # print(f"w = {w}, h = {h}")


#     # Seleccionar el contorno más grande que cumpla los criterios
#     if not possible_plates:
#         print("No se detecto nada a recortar (o no hay matricula o ya esta recortada)")
#         return image, edges_border
    

#     print("Matricula detectada en la imagen")
#     x, y, w, h = max(possible_plates, key=lambda b: b[2] * b[3])  # Por área
#     # Recortar la región de interés
#     plate = image[y:y+h, x:x+w]
#     height, width = plate.shape[:2]
#     print(width/height)
#     print(f"x = {x}, y = {y}")
#     return plate, edges_border


#############################################

def segment_characters(image):

    angle, edges = detect_rotation_angle(image)
    corrected_image = correct_rotation(image, angle)

    # Corregir la perspectiva si es necesario
    corrected_perspective = correct_perspective(corrected_image)

    # cropped_plate, canny = cut_plate(image)
    normalized_plate = cv2.resize(corrected_perspective, target_size, interpolation=cv2.INTER_AREA)

    # Aplicar Otsu
    _, binary = cv2.threshold(normalized_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)    # Hace falta invertirlo para la siguiente parte

    # Etiquetado de regiones conectadas
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Filtrar regiones
    min_area = 110; max_area = 850
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

    # Mostrar los resultados (son muchos, lo hago en dos tandas)
    titles = ['Canny', 'Rotada', 'Corregida']
    images = [edges, corrected_image, corrected_perspective]
    aux.plot_images(images, titles)


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

