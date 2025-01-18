import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.regularizers import l2

# Crear modelo
model = Sequential([
    Input(shape=(28, 28)),  # Example for a 28x28 input
    Flatten(),
    # Flatten(input_shape=(28, 28)),  # Convertir matriz 28x28 en vector 784
    Dense(256, activation='relu'),  # Capa oculta con 128 neuronas
    Dropout(0.3),                   # Dropout
    Dense(128, activation='relu'),   # Capa oculta con 64 neuronas
    Dense(36, activation='softmax') # Capa de salida con 36 clases
])

# Compilar modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
