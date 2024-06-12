import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from tensorflow import keras

def categorizar(img,model):
  img = np.array(img).astype(float)/255
  img = cv2.resize(img, (224,224))
  prediccion = modelo.predict(img.reshape(-1, 224, 224, 3))
  return np.argmax(prediccion[0], axis=-1)

modelo = keras.models.load_model('modelo_RGB.keras',custom_objects={'KerasLayer': hub.KerasLayer})
print(modelo.summary())
dataset_folder_test = "datasetRGB/test"
print("Loading imgs from folder: " + dataset_folder_test)
labels = next(os.walk(dataset_folder_test))[1]
for l in labels:
    folder = dataset_folder_test + "/" + l
    print("exploring folder " + folder)
    files = next(os.walk(folder))[2]
    for f in files:
        img   = cv2.imread(folder + "/" + f)
        prediccion = categorizar(img,modelo)
        print(prediccion) 
    