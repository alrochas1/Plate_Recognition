import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from neural_network_aux import load_images

# Cargar el modelo guardado y los datos
model = tf.keras.models.load_model('./neural_model/nn_model.keras')

data_directory = "./neural_model/characters/testing_data"
X_val, Y_val = load_images(data_directory)
Y_val = to_categorical(Y_val, num_classes=36)

# Evaluar el modelo
loss, accuracy = model.evaluate(X_val, Y_val)
print(f"Pérdida en validación: {loss:.4f}")
print(f"Precisión en validación: {accuracy:.4f}")
