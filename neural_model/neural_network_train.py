from tensorflow.keras.utils import to_categorical
from neural_network import model

from aux import load_images

data_directory = "./characters/training_data"

# Cargar y procesar datos
X_train, y_train = load_images(data_directory)


# # Normalizar las imágenes
# X_train = X_train / 255.0  # X_train debe ser un array numpy con imágenes
# y_train = to_categorical(y_train, num_classes=36)  # Etiquetas one-hot

# Entrenar el modelo
y_train = to_categorical(y_train, num_classes=36)
model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, shuffle=True)
model.save("./neural_model/nn_characters.keras")

