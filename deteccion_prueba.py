import cv2
import numpy as np
from tensorflow import keras

# cargar el modelo entrenado
model = keras.models.load_model('my_model.h5')

# definir las etiquetas
labels = {0: 'non_melanoma', 1: 'melanoma'}

# iniciar la captura de video
cap = cv2.VideoCapture(0)

# mientras se está capturando video
while True:
    # capturar un cuadro del video
    ret, frame = cap.read()

    # si no se pudo capturar el cuadro, salir del bucle
    if not ret:
        break

    # redimensionar el cuadro a 128x128 píxeles
    resized_frame = cv2.resize(frame, (128, 128))

    # normalizar el cuadro
    normalized_frame = resized_frame / 255.0

    # agregar una dimensión al cuadro para que tenga la forma (1, 128, 128, 3)
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # hacer una predicción con el modelo
    prediction = model.predict(input_frame)[0]

    print("Predicción:", prediction)

    # encontrar la etiqueta correspondiente a la clase predicha
    label = labels[np.argmax(prediction)]

    # mostrar la etiqueta en el cuadro del video
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # mostrar el cuadro del video
    cv2.imshow('frame', frame)

    # salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# liberar los recursos y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
