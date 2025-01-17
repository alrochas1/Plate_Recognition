import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

# Crear modelo
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Convertir matriz 28x28 en vector 784
    Dense(128, activation='relu'),  # Capa oculta con 128 neuronas
    Dropout(0.2),                   # Dropout
    Dense(64, activation='relu'),   # Capa oculta con 64 neuronas
    Dense(36, activation='softmax') # Capa de salida con 36 clases
])

# Compilar modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
