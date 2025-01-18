from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from neural_model.neural_network import model

from neural_model.neural_network_aux import load_images


def training():

    data_directory = "./neural_model/characters/training_data"

    early_stopping = EarlyStopping(
        monitor='val_loss',        # Monitorear la pérdida de validación
        patience=50,               # Número de épocas sin mejora antes de detener
        verbose=1,                 # Mostrar información sobre el early stopping
        restore_best_weights=True  # Restaurar los mejores pesos al final del entrenamiento
    )

    # Cargar y procesar datos
    X_Train, Y_Train, X_Val, Y_Val = load_images("train", data_directory)
    Y_Train = to_categorical(Y_Train, num_classes=36)
    Y_Val = to_categorical(Y_Val, num_classes=36)

    # Entrenar el modelo
    model.fit(X_Train, Y_Train, epochs=1000, batch_size=32, validation_data=(X_Val, Y_Val), shuffle=True, callbacks=[early_stopping])
    model.save("./neural_model/nn_model.keras")
    print("Saving model ...")



