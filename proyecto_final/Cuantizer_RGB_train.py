import sys
import os
from sympy import centroid
import yaml
import cv2
import numpy as np
from sklearn import cluster
import lbg


def get_Centroid(folder):
    print("Calculating centroid for images in: " + folder)
    files = next(os.walk(folder))[2]
    x_all = np.zeros((1, 3))
    for f in files:
        img = cv2.imread(folder + "/" + f)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, dtype=np.float64)/255
        num_cluster = 8
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        xr = R.reshape((-1, 1))
        xg = G.reshape((-1, 1))
        xb = B.reshape((-1, 1))
        x = np.concatenate((xr, xg, xb), axis=1)
        x_all = np.concatenate((x_all, x), axis=0)
    filtro = np.any(x_all != 0, axis=1)
    x_all = x_all[filtro]
    # kmeans = cluster.KMeans(n_clusters=num_cluster, n_init=4)
    centroids = lbg.lbg(x_all[1:, :], 8)
    # kmeans.fit(x_all[1:, :])
    # centroids = kmeans.cluster_centers_
    # labels = kmeans.labels_
    print(centroids)
    return centroids


def main(dataset_folder):
    print("Loading imgs from folder: " + dataset_folder)
    labels = os.walk(dataset_folder)
    labels = next(labels)[1]

    # for dir, subDir, files in os.walk(dataset_folder):
    #     print(dir)

    Centroids = {}
    for l in labels:
        gC = get_Centroid(dataset_folder + "/" + l)
        gC = np.reshape(gC, (1, -1))
        centroid = [float(h) for h in gC[0, :]]
        Centroids[l] = centroid
    with open('Final_Project/Centroids_RGB.yaml', 'w') as f:
        yaml.dump(Centroids, f)


if __name__ == "__main__":
    dataset_folder = "Final_Project/Neuronal_net/datasetRGB/train" if len(
        sys.argv) < 2 else sys.argv[1]
    print("Current directory: " + os.getcwd())
    main(dataset_folder)
