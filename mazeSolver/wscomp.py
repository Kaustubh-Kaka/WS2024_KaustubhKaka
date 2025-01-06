import numpy as np
from math import *
import cv2

def dist(a, b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5

def turnangle(a, b, c):
    dvec1 = (b[0]-a[0], b[1]-a[1])
    dvec2 = (c[0]-b[0], c[1]-b[1])
    return acos((dvec1[0]*dvec2[0]+dvec1[1]*dvec2[1])/(dist((0,0),dvec1)*dist((0,0),dvec2)))


img = cv2.imread("./bigmaze_iv.png")

#image preprocessing

fac = 1
newshape = (img.shape[1]//fac,img.shape[1]//fac)

img = cv2.resize(img,newshape)

gsimg = np.zeros((img.shape[0],img.shape[1]))

thresh = 0.3

for i in range(len(img)):
    for j in range(len(img[0])):
        gsimg[i,j] = (0.3*img[i,j][0]+0.59*img[i,j][1]+0.11*img[i,j][2])/255

# gsimg = cv2.GaussianBlur(gsimg, (0,0), sigmaX=3, sigmaY=3)


for i in range(len(img)):
    for j in range(len(img[0])):
        if(gsimg[i,j]>thresh):gsimg[i,j] = int(1)
        else: gsimg[i,j] = int(0)

rgb_img = np.zeros((gsimg.shape[0],gsimg.shape[1],3), dtype=np.uint8)

for i in range(len(img)):
    for j in range(len(img[0])):
        if(gsimg[i,j]==0): rgb_img[i,j] = np.array((0,0,0))
        else: rgb_img[i,j] = np.array((255,255,255))

# preprocessing end

# insert points

numpoints = 200 # redundant since no point sampling

pointslist = []

griddim = 30 # grid dimension, assumed square, can be changed easily

cellxsize, cellysize = len(img)/griddim, len(img[0])/griddim

for i in range(griddim):
    for j in range(griddim):
        rpoint = (int(round(i*cellxsize+cellxsize/2)),int(round(j*cellysize+cellysize/2)))
        if(gsimg[rpoint[0], rpoint[1]]==0):
            pointslist.append(rpoint)

# end

def drawLine(point1,point2):  # check whether the line between two points intersects a wall
    point1 = (point1[0], point1[1])
    point2 = (point2[0], point2[1])
    slope = (point2[1]-point1[1])/(point2[0]-point1[0]+0.001)
    if((point2[0]-point1[0])==0):
        for i in range(min(point1[1],point2[1]),max(point1[1],point2[1])):
            # rgb_img[point1[0],i] = np.array((0,255,0))
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
                # rgb_img[i,j] = np.array((0,255,0))
                if(gsimg[i,j]==1): return False
        
    return True

# graph creation

adjList = []

visited = []
distance = []
par = []

for i in range(len(pointslist)):
    adjList.append([])
    visited.append(False)
    distance.append(int(1e6))
    par.append(-1)

for i in range(len(pointslist)):
    if(i>0 and drawLine(pointslist[i],pointslist[i-1])): 
        adjList[i].append(i-1)
        adjList[i-1].append(i)
        cv2.line(rgb_img,pointslist[i][::-1],pointslist[i-1][::-1],(255,0,0),1)
    if(i<len(pointslist)-1 and drawLine(pointslist[i],pointslist[i+1])): 
        adjList.append(i+1)
        adjList[i+1].append(i)
        cv2.line(rgb_img,pointslist[i][::-1],pointslist[i+1][::-1],(255,0,0),1)
    if(i>=griddim and drawLine(pointslist[i],pointslist[i-griddim])): 
        adjList[i].append(i-griddim)
        adjList[i-griddim].append(i)
        cv2.line(rgb_img,pointslist[i][::-1],pointslist[i-griddim][::-1],(255,0,0),1)
    if(i<len(pointslist)-griddim and drawLine(pointslist[i],pointslist[i+griddim])): 
        adjList[i].append(i+griddim)
        adjList[i+griddim].append(i)
        cv2.line(rgb_img,pointslist[i][::-1],pointslist[i+griddim][::-1],(255,0,0),1)

print(adjList[0],adjList[1])

# end

# djikstra to find the path

start, end = 14, 884 # start and end points in pointslist
distance[start] = 0
djikstra = [(distance[start],start)]

while(len(djikstra)>0):
    curdist,cur = djikstra.pop()
    if(visited[cur]): continue
    
    visited[cur] = True
    for x in adjList[cur]:
        if(not visited[x]):
            if(distance[cur]+1<distance[x]):
                distance[x] = min(distance[x], distance[cur]+1)
                djikstra.append((distance[x],x))
                par[x] = cur
    djikstra.sort(reverse=True)
    if(visited[end]): break

# print(visited)
# print(distance)

# djikstra end

v = end
pathlist = []

while(v!=-1): # highlighting the path itself
    pathlist.append(v)
    prev = par[v]
    if(prev==-1): break
    cv2.line(rgb_img, pointslist[prev][::-1], pointslist[v][::-1], (0,0,255),1)
    v = prev

pathlist.reverse()
print(pathlist)


# for i in range(len(pathlist)-1):
#     # if(i>0): print("turn by", (180/pi)*turnangle(pointslist[pathlist[i-1]], pointslist[pathlist[i]],pointslist[pathlist[i+1]]))
#     # print("move forward", dist(pointslist[pathlist[i]],pointslist[pathlist[i+1]]))
#     print((int(round(pointslist[pathlist[i]][0]//cellxsize)),int(round(pointslist[pathlist[i]][1]//cellysize))))


for i in range(len(pointslist)): # display all points in yellow    
    rgb_img[pointslist[i][0],pointslist[i][1]] = np.array((0,255,255))


cv2.imwrite("./prm.png", rgb_img)