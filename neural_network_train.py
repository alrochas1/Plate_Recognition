from tensorflow.keras.utils import to_categorical
from neural_network import model

# Normalizar las imágenes
X_train = X_train / 255.0  # X_train debe ser un array numpy con imágenes
y_train = to_categorical(y_train, num_classes=36)  # Etiquetas one-hot

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
