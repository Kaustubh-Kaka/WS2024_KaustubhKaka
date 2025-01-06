import numpy as np
from math import *
import cv2
import random

def dist(a, b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5

img = cv2.imread("./maze.jpeg")

#image preprocessing

fac = 3
newshape = (img.shape[0]//fac,img.shape[1]//fac)

img = cv2.resize(img,newshape)

gsimg = np.zeros((img.shape[0],img.shape[1]))

thresh = 0.3

for i in range(len(img)):
    for j in range(len(img[0])):
        gsimg[i,j] = (0.3*img[i,j][0]+0.59*img[i,j][1]+0.11*img[i,j][2])/255

gsimg = cv2.GaussianBlur(gsimg, (0,0), sigmaX=3, sigmaY=3)


for i in range(len(img)):
    for j in range(len(img[0])):
        if(gsimg[i,j]>thresh):gsimg[i,j] = int(1)
        else: gsimg[i,j] = int(0)

rgb_img = np.zeros((gsimg.shape[0],gsimg.shape[1],3), dtype=np.uint8)

for i in range(len(img)):
    for j in range(len(img[0])):
        if(gsimg[i,j]==0): rgb_img[i,j] = np.array((0,0,0))
        else: rgb_img[i,j] = np.array((255,255,255))

#preprocessing end

# randomly sample points

actualStart = (211, 414) # pixel coordinates close to start and end
actualEnd = (211,12)

numpoints = 200

pointslist = [actualStart, actualEnd]

for i in range(numpoints):
    rpoint = (random.randint(0,len(rgb_img)-2),(random.randint(0,len(rgb_img[0])-2)))
    if(gsimg[rpoint[0], rpoint[1]]==0):
        pointslist.append(rpoint)

# end random sampling

def drawLine(point1,point2):  # check whether the line between two points intersects a wall
    slope = (point2[1]-point1[1])/(point2[0]-point1[0]+0.001)
    if((point2[0]-point1[0])==0):
        for i in range(min(point1[1],point2[1]),max(point1[1],point2[1])):
            # rgb_img[point1[0],i] = np.array((0,0,255))
            if(gsimg[point1[0],i]==1): return False
    else:
        xmin,ymin,xmax = 0,0,0
        if(point1[0]<point2[0]):
            (xmin,ymin) = point1
            xmax = point2[0]
        else:
            (xmin,ymin) = point2
            xmax = point1[0]
        for i in range(xmin,xmax):
            jmin = min(int(round(slope*(i-xmin))),int(round(slope*(i+1-xmin))))
            jmax = max(int(round(slope*(i-xmin))),int(round(slope*(i+1-xmin))))
            for j in range(ymin+jmin,ymin+jmax+1):
                # rgb_img[i,j] = np.array((0,0,255))
                if(gsimg[i,j]==1): return False
            # print((i,point1[1]+int(round(slope*(i-point1[0])))))
    return True

# PRM graph creation

kcloset = 10 # number of closest points considered for PRM

adjList = []

visited = []
distance = []
par = []

for i in range(len(pointslist)):
    adjList.append([])
    visited.append(False)
    distance.append(1e6)
    par.append(-1)

for i in range(len(pointslist)):
    dlist = []
    for j in range(len(pointslist)):
        if(j!=i and drawLine(pointslist[i],pointslist[j])):
            dlist.append((dist(pointslist[i],pointslist[j]),j))
    dlist.sort()
    for j in range(len(dlist)):
        cv2.line(rgb_img,pointslist[i][::-1],pointslist[dlist[j][1]][::-1],(255,0,0),1)
        adjList[i].append(dlist[j][1])
        if(len(adjList[i])>=kcloset): break

# PRM end

# djikstra to find the path

start, end = 0, 1 # start and end points in pointslist
distance[start] = 0
djikstra = [(distance[start],start)]

while(len(djikstra)>0):
    curdist,cur = djikstra.pop()
    if(visited[cur]): continue
    
    visited[cur] = True
    for x in adjList[cur]:
        if(not visited[x]):
            if(distance[cur]+dist(pointslist[cur],pointslist[x])<distance[x]):
                distance[x] = min(distance[x], distance[cur]+dist(pointslist[cur],pointslist[x]))
                djikstra.append((distance[x],x))
                par[x] = cur
    djikstra.sort(reverse=True)
    if(visited[end]): break

v = end

# djikstra end

while(v!=-1): # highlighting the path itself
    prev = par[v]
    if(prev==-1): break
    cv2.line(rgb_img, pointslist[prev][::-1], pointslist[v][::-1], (0,0,255),1)
    v = prev

for x in adjList[0]:
    cv2.line(rgb_img, pointslist[x][::-1], actualStart[::-1], (255,255,0),1)
for x in adjList[1]:
    cv2.line(rgb_img, pointslist[x][::-1], actualEnd[::-1], (255,255,0),1)

for i in range(len(pointslist)): # display all points in yellow
    if(i>1):
        rgb_img[pointslist[i][0],pointslist[i][1]] = np.array((0,255,255))

rgb_img[actualStart[0],actualStart[1]] = np.array((255,0,255)) # endpoints in green
rgb_img[actualEnd[0],actualEnd[1]] = np.array((255,0,255))

cv2.imwrite("./prm.png", rgb_img)