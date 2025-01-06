import numpy as np
import cv2
# from math import *

fname = input("(k-means image posterization)Please enter file name: ")

img = cv2.imread(fname)

fac = 6 # scales each dimension of image by factor(images are big and python is slow)
newshape = (img.shape[1]//fac,img.shape[0]//fac)

img = cv2.resize(img,newshape)

# try playing around with these
k = 7 # number of colors
it = 10 # number of clustering iterations

centroids = np.zeros((k,3))
for i in range(k):
    centroids[i] = (np.random.randint(0,256,size=(3)))

newcentroids = np.zeros((k,3))
cnt = np.zeros(k)

width, height = len(img), len(img[0])


for curit in range(it):
    for i in range(width):
        for j in range(height):
            md = 1e6
            mind = 0
            d = 0
            for l in range(k):
                d = np.dot(img[i,j]-centroids[l],img[i,j]-centroids[l])
                if(d<md):
                    md = d
                    mind = l
            newcentroids[mind]+=img[i,j]
            cnt[mind]+=1
    # print(cnt)
    # print("a",newcentroids)
    for j in range(k):
        for k1 in range(3):
            if(cnt[j]):
                newcentroids[j][k1]//=cnt[j]
    # print("b",newcentroids)
    centroids = newcentroids

    newcentroids = np.zeros((k,3))
    cnt = np.zeros(k)
    print(str(100*(curit+1)/it)+"% done")

# print(centroids)

for i in range(width):
        for j in range(height):
            md = 1e6
            mind = 0
            d = 0
            for l in range(k):
                d = np.dot(img[i][j]-centroids[l],img[i][j]-centroids[l])
                if(d<md):
                    md = d
                    mind = l
            img[i][j] = centroids[mind]

cv2.imwrite(fname.split('.')[0]+"_mod.png",img)