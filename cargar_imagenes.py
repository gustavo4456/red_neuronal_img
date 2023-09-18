import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Definir las rutas a las carpetas de imágenes de cada clase
melanoma_path = "/home/gustavo/Imágenes/maligno"
non_melanoma_path = "/home/gustavo/Imágenes/beningno"

# Definir el tamaño de las imágenes a cargar
img_width, img_height = 100, 100

# Función para aplicar las mejoras
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

# Función para cargar las imágenes y procesarlas
def load_and_preprocess_images(folder, num_images):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.JPG')) and len(images) < num_images:
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.resize(img, (img_width, img_height))
                processed_img = apply_image_enhancements(img)
                images.append(processed_img)
    return np.array(images)

# Cargar y procesar las imágenes de las dos carpetas en matrices NumPy separadas
num_images_per_class = 10
melanoma_images = load_and_preprocess_images(melanoma_path, num_images_per_class)
non_melanoma_images = load_and_preprocess_images(non_melanoma_path, num_images_per_class)

# Guardar las matrices NumPy en archivos .npy
np.save('melanoma_processed.npy', melanoma_images)
np.save('non_melanoma_processed.npy', non_melanoma_images)

# Mostrar la cantidad de imágenes en cada categoría
print("Cantidad de imágenes de melanoma:", len(melanoma_images))
print("Cantidad de imágenes de no melanoma:", len(non_melanoma_images))


# Mostrar dos imágenes de melanoma
num_images_to_show = 10
for i in range(num_images_to_show):
    plt.subplot(2, num_images_to_show, i+1)
    plt.imshow(cv2.cvtColor(melanoma_images[i], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Melanoma')
    
# Mostrar dos imágenes de no melanoma
for i in range(num_images_to_show):
    plt.subplot(2, num_images_to_show, num_images_to_show + i + 1)
    plt.imshow(cv2.cvtColor(non_melanoma_images[i], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('No Melanoma')

plt.tight_layout()
plt.show()