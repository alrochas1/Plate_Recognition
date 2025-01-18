import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from neural_model.neural_network_aux import load_images

def testing():
    # Cargar el modelo guardado y los datos
    model = tf.keras.models.load_model('./neural_model/nn_model.keras')

    data_directory = "./neural_model/characters/testing_data"
    X_Test, Y_Test = load_images("test", data_directory)
    Y_Test = to_categorical(Y_Test, num_classes=36)

    # Evaluar el modelo
    loss, accuracy = model.evaluate(X_Test, Y_Test)
    print(f"Pérdida en validación: {loss:.4f}")
    print(f"Precisión en validación: {accuracy:.4f}")
