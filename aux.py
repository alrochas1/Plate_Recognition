import matplotlib.pyplot as plt
import numpy as np
import cv2

from neural_model.neural_network import res

def plot_images(images, titles):

    plt.figure(figsize=(15, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()



# Para normalizar los caracteres sin deformarlos
def normalize(char_img, size=(res, res)):

    # Mantener proporciones originales
    h, w = char_img.shape
    aspect_ratio = w / h
    
    if aspect_ratio > 1:  # Más ancho que alto
        new_w = size[1]
        new_h = int(size[1] / aspect_ratio)
    else:                 # Más alto que ancho
        new_h = size[0]
        new_w = int(size[0] * aspect_ratio)
    
    if (new_h == 0) or (new_h == 0):
        print("La imagen es demasiado alargada (no deberia haber pasado los filtros previos)")
        char = cv2.resize(char_img, size, interpolation=cv2.INTER_AREA)
        return char
    else:
        resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Crear lienzo blanco (negro para luego invertirlo) y centrar el carácter
        normalized = np.ones(size, dtype=np.uint8) * 0
        y_offset = (size[0] - new_h) // 2
        x_offset = (size[1] - new_w) // 2
        normalized[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return normalized