import cv2
import numpy as np
from math import *

# preprocessing

img = cv2.imread("maze.jpeg")

fac = 3

img = cv2.resize(img,(len(img)//3, len(img)//3))

gsimg = np.zeros((img.shape[0],img.shape[1]))

thresh = 0.3

for i in range(len(img)):
    for j in range(len(img[0])):
        gsimg[i,j] = (0.3*img[i,j][0]+0.59*img[i,j][1]+0.11*img[i,j][2])/255

gsimg = cv2.GaussianBlur(gsimg, (0,0), sigmaX=3, sigmaY=3)

rgb_img = np.zeros((img.shape[0], img.shape[1], 3))

for i in range(len(img)):
    for j in range(len(img[0])):
        if(gsimg[i,j]>thresh):
            rgb_img[i,j] = np.array((255,255,255))
        else: rgb_img[i,j] = np.array((0,0,0))


cv2.imwrite("./rrt.png", rgb_img)

# end