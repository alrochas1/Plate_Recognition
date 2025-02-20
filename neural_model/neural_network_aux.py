import os
import numpy as np
import cv2
import random

import aux
from neural_model.neural_network import res
# import matplotlib.pyplot as plt


def process_image(image):

    median = cv2.medianBlur(image, 3)

    # Aplicar Otsu
    _, binary = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Esto no es muy eficiente pero funciona
    binary = cv2.bitwise_not(binary)
    char = aux.normalize(binary)
    char = cv2.bitwise_not(char)

    return char


def load_images(mode, data_dir, target_size=(res, res)):

    X = []
    Y = []
    class_labels = sorted(os.listdir(data_dir))  # Lista ordenada de carpetas (clases)
    label_map = {label: idx for idx, label in enumerate(class_labels)}

    for label in class_labels:
        folder_path = os.path.join(data_dir, label)
        if os.path.isdir(folder_path):  # Asegurarse de que sea una carpeta
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_processed = process_image(img)
                    img_resized = cv2.resize(img_processed, target_size, interpolation=cv2.INTER_AREA)
                    img_normalized = img_resized / 255.0
                    X.append(img_normalized)
                    Y.append(label_map[label])
                    # titles = ['Original', 'Procesada']
                    # images = [img, img_normalized]
                    # aux.plot_images(images, titles)
                    # plt.show()

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)



    if mode == "test":
        print("Cargando Datos de Prueba ...")
        return X, Y
    
    else:
        # Dividir los datos por clase para evitar sesgo en la distribución
        X_Train = []
        Y_Train = []
        X_Val = []
        Y_Val = []

        # Recorrer cada clase para dividir sus datos
        for label in class_labels:
            class_idx = label_map[label]
            class_indices = np.where(Y == class_idx)[0]
            X_class = X[class_indices]
            Y_class = Y[class_indices]
            
            # Mezclar los datos de manera aleatoria
            combined = list(zip(X_class, Y_class))
            random.shuffle(combined)
            X_class, Y_class = zip(*combined)

            # Dividir el 80% para entrenamiento y el 20% para validación
            split_index = int(len(X_class) * 0.8)
            X_class_train, X_class_val = X_class[:split_index], X_class[split_index:]
            Y_class_train, Y_class_val = Y_class[:split_index], Y_class[split_index:]

            X_Train.append(X_class_train)
            Y_Train.append(Y_class_train)
            X_Val.append(X_class_val)
            Y_Val.append(Y_class_val)


        X_Train = np.concatenate(X_Train, axis=0)
        Y_Train = np.concatenate(Y_Train, axis=0)
        X_Val = np.concatenate(X_Val, axis=0)
        Y_Val = np.concatenate(Y_Val, axis=0)

        print("Cargando Datos de Entrenamiento ...")
        return X_Train, Y_Train, X_Val, Y_Val