import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Función para aplicar un procesamiento de la imagen
def apply_image_enhancements(image):
    # Extraer el canal azul de la imagen
    blue_channel = image[:, :, 0]  # El canal azul está en la posición 0 del eje RGB
    
    # Aplicar un ajuste de contraste al canal azul para realzar el color
    alpha = 1.5  # Factor de ajuste de contraste
    beta = 20    # Factor de ajuste de brillo
    enhanced_blue_channel = np.clip(alpha * blue_channel + beta, 0, 255).astype(np.uint8)
    
    # Crear una copia de la imagen original y reemplazar el canal azul con el canal realzado
    enhanced_image = image.copy()
    enhanced_image[:, :, 0] = enhanced_blue_channel
    
    # Convertir la imagen realzada a escala de grises
    gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    
    # Mezclar la imagen en escala de grises con el canal azul realzado
    weighted_gray_image = cv2.addWeighted(gray_image, 0.5, enhanced_blue_channel, 0.5, 0)
    
    return weighted_gray_image

# cargar el modelo entrenado
model = keras.models.load_model('best_model.h5')

# definir las etiquetas
labels = {0: 'non-melanoma', 1: 'melanoma'}

# cargar una imagen de prueba
img = cv2.imread('prueba_non_melanoma.JPG')

# aplicar el procesamiento a la imagen de prueba
processed_img = apply_image_enhancements(img)

# redimensionar la imagen a 128x128 píxeles
resized_img = cv2.resize(processed_img, (100, 100))

# normalizar la imagen
normalized_img = resized_img / 255.0

# agregar una dimensión a la imagen para que tenga la forma (1, 128, 128, 3)
input_img = np.expand_dims(normalized_img, axis=0)

# Obtener la predicción del modelo
predictions = model.predict(input_img)
predicted_class = np.argmax(predictions[0])  # Índice de la clase con mayor probabilidad
class_names = ['No Melanoma', 'Melanoma']  # Nombre de las clases

# Mostrar la imagen de entrada
plt.imshow(input_img[0], cmap='gray')
plt.axis('off')
plt.title(f'Predicted Class: {class_names[predicted_class]}')
plt.show()

