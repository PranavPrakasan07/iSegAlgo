import numpy as np
from numpy import array
from PIL import Image
import datetime
import imageio

a = datetime.datetime.now()
face = imageio.imread('lion.pgm')

print(face.shape)

convx = array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

l = face.shape[0]
b = face.shape[1]

padded = np.zeros((l+2,b+2)).astype(np.uint8)

i = None
j = None

for i in range(0, l):
    for j in range(0, b):
        padded[i+1][j+1] = face[i][j]

res = np.zeros((l,b)).astype(np.uint8)

i = None
j = None

for i in range(1, l+1):
    for j in range(1, b+1):
        res[i-1][j-1] = (convx[0][0]*padded[i-1][j-1] + convx[0][1]*padded[i-1][j]+convx[0][2]*padded[i-1][j+1] +
                         convx[1][0]*padded[i][j-1]+convx[1][1]*padded[i][j] + convx[1][2]*padded[i][j+1] +
                         convx[2][0]*padded[i+1][j-1] + convx[2][1]*padded[i+1][j] + convx[2][2]*padded[i+1][j+1])

        res[i-1][j-1] = (res[i-1][j-1]**2)

resy = np.zeros((l+2,b+2)).astype(np.uint8)

i = None
j = None
convy = [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]
         ]

for i in range(1, l+1):
    for j in range(1, b+1):
        resy[i-1][j-1] = (convy[0][0]*padded[i-1][j-1] + convy[0][1]*padded[i-1][j]+convy[0][2]*padded[i-1][j+1] +
                          convy[1][0]*padded[i][j-1]+convy[1][1]*padded[i][j] + convy[1][2]*padded[i][j+1] +
                          convy[2][0]*padded[i+1][j-1] + convy[2][1]*padded[i+1][j] + convy[2][2]*padded[i+1][j+1])

        resy[i-1][j-1] = (resy[i-1][j-1]**2)

res2 = np.zeros((l,b)).astype(np.uint8)

for i in range(0, l):
    for j in range(0, b):
        res2[i][j] = int((res[i-1][j-1]+int(resy[i-1][j-1])))
        if res2[i][j] > 15:
            res2[i][j] = 255


img = Image.fromarray(res2.astype(np.uint8))
img.save('SEDS.png')
# img.show()

b = datetime.datetime.now()
print(b-a)
