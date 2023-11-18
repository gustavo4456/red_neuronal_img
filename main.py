import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos de imágenes de melanoma y no melanoma
melanoma = np.load('melanoma_processed.npy')
non_melanoma = np.load('non_melanoma_processed.npy')

# Definir cuántas imágenes deseas tomar de cada clase 7300
num_images_per_class = 7300

# Tomar un número específico de imágenes de cada clase
melanoma = melanoma[:num_images_per_class]
non_melanoma = non_melanoma[:num_images_per_class]

# Normalizar los datos al rango [0, 1]
melanoma = melanoma / 255.0
non_melanoma = non_melanoma / 255.0

# Etiquetas para las imágenes de melanoma y no melanoma (1 para melanoma, 0 para no melanoma)
melanoma_labels = np.ones(melanoma.shape[0])
non_melanoma_labels = np.zeros(non_melanoma.shape[0])

# Combinar las imágenes y etiquetas en un solo conjunto de datos
data = np.concatenate((melanoma, non_melanoma), axis=0)
labels = np.concatenate((melanoma_labels, non_melanoma_labels), axis=0)

# Mezclar los datos y dividir en conjuntos de entrenamiento y prueba
indices = np.random.permutation(data.shape[0])
data = data[indices]
labels = labels[indices]
split = int(data.shape[0] * 0.8)
train_data = data[:split]
train_labels = labels[:split]
test_data = data[split:]
test_labels = labels[split:]

# Crear el modelo de red neuronal
model = keras.Sequential([
    layers.InputLayer(input_shape=(100, 100, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),
    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(512, activation='relu'),
    layers.Dense(2, activation='softmax')  # capa de salida para tener dos clases
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Usar esta función de pérdida para múltiples clases
              metrics=['accuracy'])

# Agregar callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]

# Entrenar el modelo con callbacks
history = model.fit(train_data, train_labels, batch_size=32, epochs=20, validation_data=(test_data, test_labels), callbacks=callbacks)

# Graficar la precisión (accuracy) durante el entrenamiento y la validación
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Graficar la pérdida (loss) durante el entrenamiento y la validación
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Evaluar el modelo
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Guardar el modelo entrenado
model.save('my_model.h5')
