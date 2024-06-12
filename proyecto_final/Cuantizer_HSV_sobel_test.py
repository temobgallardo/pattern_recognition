import sys
import os
import yaml
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming
from scipy import signal
import cv2

def compare_with_centroids(img,archivo):
    with open(archivo, 'r') as file:
        datos = yaml.safe_load(file)
    global_distance={}
    for clave, vectors in datos.items():
        distance_min=10
        clave_distance=0
            # Convertimos el vector a una matriz de 12 dimensiones
        matriz = np.array(vectors).reshape(-1, 9)
            # comparamos r[:,0] con los centroides y tomamos el de distancia minima
        for i in range(len(img[:,0])):
            distance_min=10
            for j in range(len(matriz[:,0])):
                distance = np.linalg.norm(img[i,:]-matriz[j,:])
                #print(distance)
                if distance < distance_min:
                    distance_min = distance
            clave_distance +=distance_min     
            #print(clave_distance)
            #print(f"distancia minima entre r {i} y centroide{clave}")
        global_distance[clave] = clave_distance
        
        #print(f"terminamos con el clave {clave}")
                #comparamos r[:,1] con los centroides de 1 y tomamos el de distancia minima
                #...
                #sumar la distancia minima y almacenarla
            # comparamos r[:,0] con los centroides de 2 y tomamos el de distancia minima 
                #comparamos r[:,1] con los centroides de 2 y tomamos el de distancia minima
                #...
                #sumar la distancia minima y almacenarla
            #...
    #print(global_distance)
    label_min = min(global_distance, key=global_distance.get)
    print(label_min)   
            #comparar las distancias totales de cada comparacion y asignar un numero a la imagen con la distancia minima

   
def main(dataset_folder):
    print("Loading images from folder: " + dataset_folder)
    labels = next(os.walk(dataset_folder))[1]
    for l in labels:
        folder = dataset_folder + "/" + l
        files = next(os.walk(folder))[2]
        for f in files:
            img  = cv2.imread(folder + "/" + f)
            print(f"file  {folder}")
            if img is None:
                continue
            sobel_x_image = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
            sobel_y_image = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
            sobel_x = cv2.convertScaleAbs(sobel_x_image)/255
            sobel_y = cv2.convertScaleAbs(sobel_y_image)/255
            img = np.asarray(img,dtype=np.float64)/255
            R = img[:,:,0]
            G = img[:,:,1]
            B = img[:,:,2]
            SXR = sobel_x[:,:,0]
            SXG = sobel_x[:,:,1]
            SXB = sobel_x[:,:,2]
            SYR = sobel_y[:,:,0]
            SYG = sobel_y[:,:,1]
            SYB = sobel_y[:,:,2]
            r = R.reshape((-1,1))
            g = G.reshape((-1,1))
            b = B.reshape((-1,1))
            xr = SXR.reshape((-1,1))
            xg = SXG.reshape((-1,1))
            xb = SXB.reshape((-1,1))
            yr = SYR.reshape((-1,1))
            yg = SYG.reshape((-1,1))
            yb = SYB.reshape((-1,1))
            x = np.concatenate((r,g,b,xr,xg,xb,yr,yg,yb),axis = 1)
            filtro = np.any(x !=0,axis=1)
            x = x[filtro]
            compare_with_centroids(x,"Centroids_HSV_sobel.yaml")

if __name__ == "__main__":
    dataset_folder = "Neuronal_net/datasetHSV/test" if len(sys.argv) < 2 else sys.argv[1]
    main(dataset_folder)