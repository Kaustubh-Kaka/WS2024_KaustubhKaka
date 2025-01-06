import numpy as np
from scipy import ndimage
from math import *
import cv2

def clamp(l,r,x):
    if(x<l):return l
    elif(x>r): return r
    else: return x

def hsl_to_rgb(h, s, l):
    def hue_to_rgb(p, q, t):
        t += 1 if t < 0 else 0
        t -= 1 if t > 1 else 0
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: p + (q - p) * (2/3 - t) * 6
        return p

    if s == 0:
        r, g, b = l, l, l
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)

    return np.array((int(round(255*r)), int(round(255*g)), int(round(255*b))), dtype=np.uint8)

img = cv2.imread("./table.png")

# img = cv2.resize(img,np.array(img.shape[::-1][1:], dtype=np.uint32)//4) #image downsizing

gsimg = np.zeros(img.shape[:2])

gkernel3 = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
gkernel5 = np.array([[ 0.002969,  0.013306,  0.021938,  0.013306,  0.002969],
       [ 0.013306,  0.059634,  0.09832 ,  0.059634,  0.013306],
       [ 0.021938,  0.09832 ,  0.162103,  0.09832 ,  0.021938],
       [ 0.013306,  0.059634,  0.09832 ,  0.059634,  0.013306],
       [ 0.002969,  0.013306,  0.021938,  0.013306,  0.002969]])

fkernel = np.ones((3,3))/9

gsimg = ndimage.convolve(gsimg,gkernel3, mode='constant',cval=0.0)

for i in range(len(img)):
    for j in range(len(img[0])):
        gsimg[i,j] = (0.3*img[i,j][0]+0.59*img[i,j][1]+0.11*img[i,j][2])/255

kernel1 = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernel2 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])

skernel1 = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
skernel2 = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

lkernel1 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
lkernel2 = np.array([[1,1,1],[1,-8,1],[1,1,1]])


def sobel(img, gsmoothing=gkernel3, thresh=0):
    img = ndimage.convolve(img, gsmoothing, mode='constant', cval=1.0)
    ximg = ndimage.convolve(img, skernel1, mode='constant', cval=1.0)
    yimg = ndimage.convolve(img, skernel2, mode='constant', cval=1.0)
    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i,j] = (ximg[i,j]**2+yimg[i,j]**2)**0.5
            if img[i,j]<thresh: img[i,j] = 0
    return img

def seriesConvolve(img, klist):
    for i in range(len(klist)):
        img = ndimage.convolve(img, klist[i], mode='constant', cval=1.0)
    return img

def maxSupression(img, gsmoothing=gkernel3, thresh=0):
    img = ndimage.convolve(img, gsmoothing, mode='constant', cval=1.0)
    ximg = ndimage.convolve(img, skernel1, mode='constant', cval=1.0)
    yimg = ndimage.convolve(img, skernel2, mode='constant', cval=1.0)
    fac = 2
    closeness = 1
    dir = 0
    thresh1 = 0
    for i in range(1,len(img)-1):
        for j in range(1,len(img[0])-1):
            img[i,j] = (ximg[i,j]**2+yimg[i,j]**2)**0.5
            if img[i,j]<thresh: img[i,j] = 0

    for i in range(1,len(img)-1):
        for j in range(1,len(img[0])-1):
            if(img[i,j]>thresh1):
                dir = (round(4*atan(yimg[i,j]/(ximg[i,j]+0.001))/pi)+2)%4
                if dir==0:
                    if(img[i,j]<closeness*img[i,j-1] or img[i,j]<closeness*img[i,j+1]): img[i,j] = round(img[i,j]/fac)
                    else: img[i,j] = round(min(fac*img[i,j],255))
                elif dir==1:
                    if(img[i,j]<closeness*img[i-1,j+1] or img[i,j]<closeness*img[i+1,j-1]): img[i,j] = round(img[i,j]/fac)
                    else: img[i,j] = round(min(fac*img[i,j],255))
                elif dir==2:
                    if(img[i,j]<closeness*img[i+1,j] or img[i,j]<closeness*img[i-1,j]): img[i,j] = round(img[i,j]/fac)
                    else: img[i,j] = round(min(fac*img[i,j],255))
                else :
                    if(img[i,j]<closeness*img[i-1,j-1] or img[i,j]<closeness*img[i+1,j+1]): img[i,j] = round(img[i,j]/fac)
                    else: img[i,j] = round(min(fac*img[i,j],255))
            else:
                img[i,j] = 0
    high = 0.9
    low =0.1
    for i in range(1,len(img)-1):
        for j in range(1,len(img[0])-1):
            if img[i,j]>high:
                continue
            elif img[i,j]<low:
                img[i,j] = 0 #round(min(fac*img[i,j],255))
            else :
                nc = 0
                for k in range(3):
                    for l in range(3):
                        if(img[i+k-1,j+l-1]>high):nc+=1
                if(nc<1):
                    img[i,j] = 0

    print(img[10,10])

    return img

x = maxSupression(gsimg)
y = np.zeros((x.shape[0],x.shape[1],3),dtype=np.uint8)

for i in range(len(x)):
    for j in range(len(x[0])):
        y[i,j] = np.array( ( np.uint8(clamp(0,255,round(255*x[i,j]))),np.uint8(clamp(0,255,round(255*x[i,j]))),np.uint8(clamp(0,255,round(255*x[i,j]))) ) )


print(img[0,0],y[0,0])

imga = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
lines = cv2.HoughLinesP(
            imga, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=100, # Min number of votes for valid line
            minLineLength=5, # Min allowed length of line
            maxLineGap=10 # Max allowed gap between line for joining them
            )
imgb = np.zeros(img.shape)
lines_list = []

for points in lines:
      # Extracted points nested in the list
    x1,y1,x2,y2=points[0]
    # Draw the lines joing the points
    # On the original image
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    # Maintain a simples lookup list for points
    lines_list.append([(x1,y1),(x2,y2)])

cv2.imshow("hello", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
