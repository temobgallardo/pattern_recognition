import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
#Aumento de datos con ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2


#Crear el dataset generador
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range = 30,
    width_shift_range = 0.25,
    height_shift_range = 0.25,
    shear_range = 15,
    zoom_range = [0.5, 1.5],
    validation_split=0.2 #20% para pruebas
)

#Generadores para sets de entrenamiento y pruebas
data_gen_entrenamiento = datagen.flow_from_directory('datasetHSV/train', target_size=(224,224),
                                                     batch_size=25, shuffle=True, subset='training')
data_gen_pruebas = datagen.flow_from_directory('datasetHSV/train', target_size=(224,224),
                                                     batch_size=25, shuffle=True, subset='validation')
url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
mobilenetv2 = hub.KerasLayer(url, input_shape=(224,224,3))
#Congelar el modelo descargado
mobilenetv2.trainable = False
modelo = tf.keras.Sequential([
    mobilenetv2,
    tf.keras.layers.Dense(10, activation='softmax')
])
modelo.summary()
#Compilar
modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
EPOCAS = 100

historial = modelo.fit(
    data_gen_entrenamiento, epochs=EPOCAS, batch_size=32,
    validation_data=data_gen_pruebas
)
acc = historial.history['accuracy']
val_acc = historial.history['val_accuracy']

loss = historial.history['loss']
val_loss = historial.history['val_loss']

rango_epocas = range(EPOCAS)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(rango_epocas, acc, label='Precisión Entrenamiento')
plt.plot(rango_epocas, val_acc, label='Precisión Pruebas')
plt.legend(loc='lower right')
plt.title('Precisión de entrenamiento y pruebas')

plt.subplot(1,2,2)
plt.plot(rango_epocas, loss, label='Pérdida de entrenamiento')
plt.plot(rango_epocas, val_loss, label='Pérdida de pruebas')
plt.legend(loc='upper right')
plt.title('Pérdida de entrenamiento y pruebas')
plt.show()

modelo.save('modelo_HSV.keras')
modelosaved = keras.models.load_model('modelo_HSV.keras',custom_objects={'KerasLayer': hub.KerasLayer})
print(modelosaved.summary())
    