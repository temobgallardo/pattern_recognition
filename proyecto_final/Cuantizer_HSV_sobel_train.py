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
        sobel_x_image = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
        sobel_y_image = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
        sobel_x = cv2.convertScaleAbs(sobel_x_image)/255
        sobel_y = cv2.convertScaleAbs(sobel_y_image)/255
        img = np.asarray(img,dtype=np.float64)/255
        num_cluster = 8
        H = img[:,:,0]
        S = img[:,:,1]
        V = img[:,:,2]
        SXH = sobel_x[:,:,0]
        SXS = sobel_x[:,:,1]
        SXV = sobel_x[:,:,2]
        SYH = sobel_y[:,:,0]
        SYS = sobel_y[:,:,1]
        SYV = sobel_y[:,:,2]
        h = H.reshape((-1,1))
        s = S.reshape((-1,1))
        v = V.reshape((-1,1))
        xh = SXH.reshape((-1,1))
        xs = SXS.reshape((-1,1))
        xv = SXV.reshape((-1,1))
        yh = SYH.reshape((-1,1))
        ys = SYS.reshape((-1,1))
        yv = SYV.reshape((-1,1))
        x = np.concatenate((h,s,v,xh,xs,xv,yh,ys,yv),axis = 1)
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
    with open('Centroids_HSV_sobel.yaml','w') as f:
        yaml.dump(Centroids, f)
     
    
if __name__ == "__main__":
    dataset_folder = "Neuronal_net/datasetHSV/train" if len(sys.argv) < 2 else sys.argv[1]
    main(dataset_folder)
