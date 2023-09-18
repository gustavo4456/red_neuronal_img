import cv2

# crear objeto de captura de video
cap = cv2.VideoCapture(0)

while True:
    # leer el siguiente frame de la cámara
    ret, frame = cap.read()

    # mostrar el frame en una ventana
    cv2.imshow('frame', frame)

    # salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# liberar el objeto de captura de video y cerrar la ventana
cap.release()
cv2.destroyAllWindows()

