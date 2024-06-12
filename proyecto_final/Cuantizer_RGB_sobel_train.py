import sys
import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming
from scipy import signal
from sklearn import cluster


def get_Centroid(folder):
    print("Calculating centroid for images in: " + folder)
    files = next(os.walk(folder))[2]
    x_all = np.zeros((1,9))
    for f in files:
        img   = cv2.imread(folder + "/" + f)
        if img is None:
            continue
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        sobel_x_image = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
        sobel_y_image = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
        sobel_x = cv2.convertScaleAbs(sobel_x_image)/255
        sobel_y = cv2.convertScaleAbs(sobel_y_image)/255
        img = np.asarray(img,dtype=np.float64)/255
        num_cluster = 8
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
        x_all = np.concatenate((x_all,x),axis=0)
    filtro = np.any(x_all !=0,axis=1)
    x_all = x_all[filtro]
    kmeans = cluster.KMeans(n_clusters=num_cluster,n_init=4)
    kmeans.fit(x_all)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    print(centroids)
    return centroids
   
def main(dataset_folder):
    print("Loading imgs from folder: " + dataset_folder)
    labels = next(os.walk(dataset_folder))[1]
    Centroids = {}
    for l in labels:
        gC = get_Centroid(dataset_folder + "/" + l)
        gC = np.reshape(gC,(1,-1))
        centroid = [float(h) for h in gC[0,:]]
        Centroids[l] = centroid
    with open('Centroids_RGB_sobel.yaml','w') as f:
        yaml.dump(Centroids, f)
     
    
if __name__ == "__main__":
    dataset_folder = "Neuronal_net/datasetRGB/train" if len(sys.argv) < 2 else sys.argv[1]
    main(dataset_folder)
