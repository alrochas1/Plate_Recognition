import cv2
import numpy as np
import matplotlib.pyplot as plt

def correct_perspective(image):
    # Detectar bordes para encontrar el contorno de la matrícula
    edges = cv2.Canny(image, 50, 150)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Seleccionar el contorno más grande (asumiendo que es la matrícula)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Aproximar el contorno a un cuadrilátero
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) != 4:
        raise ValueError("No se pudo encontrar una matrícula con 4 esquinas. Ajusta los parámetros.")
    
    # Ordenar los puntos para perspectiva (top-left, top-right, bottom-right, bottom-left)
    rect = np.zeros((4, 2), dtype="float32")
    s = approx.sum(axis=2)
    rect[0] = approx[np.argmin(s)]  # Top-left
    rect[2] = approx[np.argmax(s)]  # Bottom-right
    diff = np.diff(approx, axis=-1)
    rect[1] = approx[np.argmin(diff)]  # Top-right
    rect[3] = approx[np.argmax(diff)]  # Bottom-left

    # Establecer la nueva perspectiva
    (tl, tr, br, bl) = rect
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = max(int(height_left), int(height_right))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Obtener la transformación de perspectiva
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped