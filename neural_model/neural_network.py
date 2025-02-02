import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.regularizers import l2


res = 28    # Resolucion del modelo

# Crear modelo
model = Sequential([
    Input(shape=(res, res)), 
    Flatten(),
    Dense(512, activation='relu'),  # Capa oculta con 256 neuronas
    Dropout(0.3),                   # Dropout
    Dense(256, activation='relu'),   # Capa oculta con 128 neuronas
    Dense(36, activation='softmax') # Capa de salida con 36 clases
])

# Compilar modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
