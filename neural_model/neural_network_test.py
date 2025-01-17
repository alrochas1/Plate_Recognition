import tensorflow as tf

# Cargar el modelo guardado y los datos
model = tf.keras.models.load_model('nn_characters.keras')

data_directory = "./characters/testing_data"
X_train, y_train = load_images(data_directory)

# Normalizar los datos de validación
X_val = X_val / 255.0  # Normalizar imágenes
X_val = X_val.reshape(-1, 28, 28, 1)  # Asegurarse de que tengan la forma adecuada

# Evaluar el modelo
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Pérdida en validación: {loss:.4f}")
print(f"Precisión en validación: {accuracy:.4f}")
