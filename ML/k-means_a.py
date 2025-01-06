import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("./poscluster.png")

width,height = len(img), len(img[0])

imga = np.zeros((width,height))

for i in range(width):
    for j in range(height):
        imga[i,j] = img[i,j][0]


k = 3
it = 10

centroids = []
for _ in range(k): centroids.append(np.random.randint(0,[width, height]))
# centroid = [np.array()]

centroids = np.array(centroids)

newcentroids = np.zeros((k,2))
cnt = np.zeros(k)

print(centroids)

tempd = 0

col = np.array((np.array((255,0,0)),np.array((0,255,0)),np.array((0,0,255))))
img2 = np.zeros((width,height,3))

for _ in range(it):
    for i in range(width):
        for j in range(height):
            if(imga[i,j]):
                mind = 0
                d = 1e6
                for l in range(k):
                    tempd = np.dot(np.array((i,j))-centroids[l],np.array((i,j))-centroids[l])
                    if(tempd<d):
                        d = tempd
                        mind = l
                newcentroids[mind]+=np.array((i,j))
                cnt[mind]+=1
                
    # print(cnt)
    # print(newcentroids)
    for i in range(k):
        if(cnt[i]):
            newcentroids[i]//=cnt[i]
    centroids = newcentroids
    # print(centroids)
    newcentroids = np.zeros((k,2))
    cnt = np.zeros(k)

    
for i in range(width):
        for j in range(height):
            if(imga[i,j]):
                mind = 0
                d = 1e6
                for l in range(k):
                    tempd = np.dot(np.array((i,j))-centroids[l],np.array((i,j))-centroids[l])
                    if(tempd<d):
                        d = tempd
                        mind = l
                img2[i,j] = col[mind]
for i in range(len(centroids)):
    img2[int(centroids[i][0]),int(centroids[i][1])] = np.array((255,255,255))
    # img2 = cv2.resize(img2, (8*img2.shape[0],8*img2.shape[1]))
plt.imshow(img2)
plt.show()
# time.sleep(1000)
plt.cla()