import argparse
from PyQt5 import QtCore
import numpy as np
from scipy import misc
from sklearn import cluster
import matplotlib.pyplot as plt
import cv2

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compress the input')
    parser.add_argument("--input-file",dest="input_file",required=True,help="Input image")
    parser.add_argument("--num-bits",dest="num_bits",required=False,type=int,help="Number of bits of each pixel")
    return parser

def compress_image(img,num_cluster):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    xr = R.reshape((-1,1))
    xg = G.reshape((-1,1))
    xb = B.reshape((-1,1))
    x = np.concatenate((xr,xg,xb),axis = 1)
    print(x.shape)
    kmeans = cluster.KMeans(n_clusters=num_cluster,n_init=4)
    kmeans.fit(x)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    print(centroids.shape)
    m = xr.shape
    for i in range(m[0]):
        xr[i] = centroids[labels[i]][0]
        xg[i] = centroids[labels[i]][1]
        xb[i] = centroids[labels[i]][2]
    xr.shape = R.shape
    xg.shape = G.shape
    xb.shape = B.shape
    xr = xr[:,:,np.newaxis]
    xg = xg[:,:,np.newaxis]
    xb = xb[:,:,np.newaxis]
    input_image_compressed = np.concatenate((xr,xg,xb),axis=2)
    return input_image_compressed
def plot_image(img, title):
    vmin = img.min()
    vmax = img.max()
    plt.figure()
    plt.title(title)
    plt.imshow(img,cmap=plt.cm.gray,vmin=vmin,vmax=vmax)

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    input_file = args.input_file
    num_bits = args.num_bits
    if not 1<=num_bits<=8:
        raise TypeError('Number of bits should be between 1 and 8')
    num_cluster = np.power(2,num_bits)
    compression_rate = round(100*(8.0 - args.num_bits)/8.0,2)
    input_image = cv2.imread(input_file,1).astype(np.uint8)
    input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
    input_image = np.asarray(input_image,dtype=np.float64)/255
    plot_image(input_image,'original image')
    input_image_compressed = compress_image(input_image,20)
    plot_image(input_image_compressed,'compresed image rate'+str(compression_rate)+'%')
    plt.show()

