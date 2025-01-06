import numpy as np
import cv2
from math import *
from scipy.ndimage import rotate

# def rotp(ind,piv,angle):
#     a = complex(ind[0]-piv[0], ind[1]-piv[1])
#     a*=complex(cos(angle), sin(angle))
#     return (piv[0]+round(a.real), round(piv[1]+a.imag))

# def rot(a, pivot,angle):
#     temp = (0,0)
#     b = np.zeros((2*a.shape[0], 2*a.shape[1], 3),dtype=np.uint8)
#     for i in range(len(a)):
#         for j in range(len(a[i])):
#             temp = rotp((i,j),pivot,angle)
#             if((temp[0]+len(a))>=0 and (temp[0]+len(a))<len(b) and (temp[1]+len(a[0]))>=0 and (temp[1]+len(a[0]))<len(b[0]) ):
#                 b[temp[0]+len(a)][temp[1]+len(a[0])] = a[i][j]
#     return b


# image = cv2.imread("./images.jpeg")

# newimg = np.zeros((len(image), len(image[0])), dtype=np.uint8)

# for i in range(len(image)):
#     for j in range(len(image[i])):
#         newimg[i][j] = np.uint8((0.30*image[i][j][0]+0.59*image[i][j][1]+0.11*image[i][j][2]))

# # image = rotate(image, angle=45, reshape=False)

# # print(newimg)

# angle = 15

# cv2.imshow("hello", rot(image,(0,0), angle*pi/180))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread("./coin.jpg")

# msk = np.zeros((len(img), len(img[0])))

# for i in range(len(img)):
#     for j in range(len(img[0])):
#         if(0.30*img[i][j][0]+0.59*img[i][j][1]+0.11*img[i][j][2]>7):
#             msk[i][j] = 1

# cv2.imshow("hello", msk)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

