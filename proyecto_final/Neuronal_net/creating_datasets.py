import os
import cv2
import numpy as np
import tensorflow as tf


# Crear una función que defina un kernel personalizado
def custom_kernel_9_to_3():
    # Crear un kernel de 3x3 con 9 canales de entrada y 3 canales de salida
    # Este kernel tendrá dimensiones (kernel_height, kernel_width, in_channels, out_channels)
    kernel = np.zeros((3, 3, 9, 3), dtype=np.float32)

    # Definir las conexiones: canal 1 con 4 con 7 para el primer canal de salida
    kernel[:, :, 0, 0] = 1/3  # Canal 1 de entrada a canal 1 de salida
    kernel[:, :, 3, 0] = 1/3  # Canal 4 de entrada a canal 1 de salida
    kernel[:, :, 6, 0] = 1/3  # Canal 7 de entrada a canal 1 de salida

    # Canal 2 con 5 con 8 para el segundo canal de salida
    kernel[:, :, 1, 1] = 1/3  # Canal 2 de entrada a canal 2 de salida
    kernel[:, :, 4, 1] = 1/3  # Canal 5 de entrada a canal 2 de salida
    kernel[:, :, 7, 1] = 1/3  # Canal 8 de entrada a canal 2 de salida

    # Canal 3 con 6 con 9 para el tercer canal de salida
    kernel[:, :, 2, 2] = 1/3  # Canal 3 de entrada a canal 3 de salida
    kernel[:, :, 5, 2] = 1/3  # Canal 6 de entrada a canal 3 de salida
    kernel[:, :, 8, 2] = 1/3  # Canal 9 de entrada a canal 3 de salida

    return kernel

# Crear un modelo de Keras con una sola capa convolucional que use el kernel personalizado
def create_model(kernel):
    # Definir la capa convolucional
    conv_layer = tf.keras.layers.Conv2D(
        filters=3, kernel_size=(3, 3), padding='same', use_bias=False,
        kernel_initializer=tf.constant_initializer(kernel)
    )

    # Crear el modelo secuencial
    model = tf.keras.Sequential([conv_layer])

    # Congelar los pesos del kernel para que no se entrenen
    model.layers[0].trainable = False

    return model

# Función para aplicar el modelo a una imagen de 9 canales
def apply_custom_kernel(image_9_channels, model):
    # Asegurarse de que la imagen tenga la forma correcta (batch_size, height, width, channels)
    image_9_channels = np.expand_dims(image_9_channels, axis=0)

    # Aplicar el modelo
    image_3_channels = model.predict(image_9_channels)

    # Remover la dimensión de batch_size
    image_3_channels = np.squeeze(image_3_channels, axis=0)

    return image_3_channels
# Directorio principal del dataset
input_directory = 'datasetHSV'
output_directory = 'datasetHSVSobel'

# Crear el directorio de salida si no existe
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Función para aplicar la transformación
def transform_image(image):
    # Ejemplo: convertir a escala de grises y aplicar un desenfoque gaussiano
    sobel_x_image = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)  # Convertir a sobel x
    sobel_y_image = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)  # Convertir a sobel x
    sobel_x = cv2.convertScaleAbs(sobel_x_image)
    sobel_y = cv2.convertScaleAbs(sobel_y_image)
    transformed_image = np.concatenate((image,sobel_x,sobel_y), axis=2)
    return transformed_image

# Iterar sobre todas las clases
for class_name in os.listdir(input_directory):
    class_input_path = os.path.join(input_directory, class_name)
    class_output_path = os.path.join(output_directory, class_name)
    
    # Crear el directorio de la clase en el directorio de salida si no existe
    if not os.path.exists(class_output_path):
        os.makedirs(class_output_path)
    
    # Iterar sobre todas las imágenes en el directorio de la clase
    for filename in os.listdir(class_input_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(class_input_path, filename)
            image = cv2.imread(filepath)
            transformed_image = transform_image(image)
            kernel = custom_kernel_9_to_3()
            model = create_model(kernel)
            # Aplicar el kernel personalizado a la imagen de 9 canales
            image_3_channels = apply_custom_kernel(transformed_image.astype(np.float64), model)
            image_3_channels = cv2.convertScaleAbs(image_3_channels)
            # Guardar la imagen transformada en el directorio de salida
            output_path = os.path.join(class_output_path, filename)
            cv2.imwrite(output_path, image_3_channels)

print("Transformaciones completadas y guardadas.")
