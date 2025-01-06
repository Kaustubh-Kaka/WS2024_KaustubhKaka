import numpy as np
from math import *
from collections import deque
import cv2

img = cv2.imread("./gtamap.jpg")

gsimg = np.zeros(img.shape[:2])

for i in range(len(img)):
    for j in range(len(img[0])):
        gsimg[i,j] = (0.3*img[i,j][0]+0.59*img[i,j][1]+0.11*img[i,j][2])/255

start = (27,241)
end = (280,335)

visited = {start}
par = {start: (-1,-1)}
bfsdist = {start: 0}

bfs = deque()

bfs.append(start)

temp = (1,1)

mv = [(-1,0),(1,0),(0,1),(0,-1)]

n,m = gsimg.shape
temp2 = (0,0)

flag = False

while(len(bfs)):
    temp = bfs.popleft()
    for i in range(len(mv)):
        temp2 = (temp[0]+mv[i][0], temp[1]+mv[i][1])
        if(temp2[0]>=0 and temp2[0]<n and temp2[1]>=0 and temp2[1]<m and gsimg[temp2[0],temp2[1]]<0.4 and (not temp2 in visited)):
            bfsdist[temp2] = bfsdist[temp]+1
            bfs.append(temp2)
            visited.add(temp2)
            par[temp2] = temp
            if(temp2==end):flag = True
    # print(len(bfs))
    if(flag):
        break

pixelWritten = {(-1,-1)}

for temp in visited:
    if(temp not in pixelWritten):
        pixelWritten.add(temp)
        while(1):
            img[temp[0],temp[1]] = np.array((0,255,0),np.uint8)
            temp = par[temp]
            
            if(temp in pixelWritten):
                break
            pixelWritten.add(temp)
    
temp = end

while(1):
        img[temp[0],temp[1]] = np.array((0,0,255),np.uint8)
        temp = par[temp]
        if(temp==(-1,-1)):
            break


cv2.imwrite("./route.png",img)