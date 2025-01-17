import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función para mostrar imágenes
def mostrar_imagen(titulo, imagen):
    plt.imshow(imagen, cmap='gray')
    plt.title(titulo)
    plt.axis('off')
    plt.show()

# Cargar imagen
imagen_original = cv2.imread('5126HVL.png')  # Sustituir por la ruta de tu imagen
mostrar_imagen('Imagen Original', cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB))

# --- 2.1 Preprocesamiento de la imagen ---
# Conversión a escala de grises
imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
mostrar_imagen('Imagen en Escala de Grises', imagen_gris)

# Mejora del contraste (ecualización del histograma)
imagen_ecualizada = cv2.equalizeHist(imagen_gris)
mostrar_imagen('Imagen con Contraste Mejorado', imagen_ecualizada)

# Filtrado y reducción de ruido
imagen_suavizada = cv2.GaussianBlur(imagen_ecualizada, (5, 5), 0)
mostrar_imagen('Imagen Suavizada', imagen_suavizada)

# --- 2.2 Segmentación ---
# Detección de bordes usando el algoritmo de Canny
bordes = cv2.Canny(imagen_suavizada, 50, 150)
mostrar_imagen('Bordes Detectados', bordes)

# Umbralización (binarización)
_, imagen_binaria = cv2.threshold(imagen_suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
mostrar_imagen('Imagen Binarizada', imagen_binaria)

# Detección de contornos
contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imagen_contornos = imagen_original.copy()
cv2.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), 2)
mostrar_imagen('Contornos Detectados', cv2.cvtColor(imagen_contornos, cv2.COLOR_BGR2RGB))

# Segmentación de región de interés (ROI)
for contorno in contornos:
    # Calcular el bounding box
    x, y, w, h = cv2.boundingRect(contorno)
    # Filtrar regiones por tamaño mínimo y proporción
    if w > 50 and h > 20:  # Ajusta estos valores según la escala de tu imagen
        roi = imagen_original[y:y+h, x:x+w]
        mostrar_imagen('Región de Interés (ROI)', cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
