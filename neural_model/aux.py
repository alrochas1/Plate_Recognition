import os
import numpy as np
import cv2

def load_images(data_dir, target_size=(28, 28)):

    X = []  # Lista para almacenar las imágenes procesadas
    Y = []  # Lista para almacenar las etiquetas
    class_labels = sorted(os.listdir(data_dir))  # Lista ordenada de carpetas (clases)
    label_map = {label: idx for idx, label in enumerate(class_labels)}  # Mapa clase -> índice

    for label in class_labels:
        folder_path = os.path.join(data_dir, label)
        if os.path.isdir(folder_path):  # Asegurarse de que sea una carpeta
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                    img_normalized = img_resized / 255.0
                    X.append(img_normalized)
                    Y.append(label_map[label])
    
    # Convertir listas a arrays numpy
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)
    return X, Y