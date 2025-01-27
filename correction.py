import cv2
import numpy as np
import matplotlib.pyplot as plt

import aux


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
    

##############################################################

def correct_image(image, show=True):

    # Girar la imagen
    angle, edges = detect_rotation_angle(image)
    corrected_image = correct_rotation(image, angle)

    # Corregir la perspectiva si es necesario
    corrected_perspective = correct_perspective(corrected_image)

    if show:
        # Mostrar los resultados
        titles = ['Canny', 'Rotada', 'Corregida']
        images = [edges, corrected_image, corrected_perspective]
        aux.plot_images(images, titles)

    return corrected_perspective