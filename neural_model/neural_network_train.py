from tensorflow.keras.utils import to_categorical
from neural_model.neural_network import model

from neural_model.neural_network_aux import load_images

def training():

    data_directory = "./neural_model/characters/training_data"

    # Cargar y procesar datos
    X_train, Y_train = load_images(data_directory)

    # Entrenar el modelo
    Y_train = to_categorical(Y_train, num_classes=36)
    model.fit(X_train, Y_train, epochs=1000, batch_size=32, validation_split=0.2, shuffle=True)
    model.save("./neural_model/nn_characters.keras")



