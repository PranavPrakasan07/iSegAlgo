import numpy as np
from numpy import array
from PIL import Image
import datetime
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb_to_gray(img):
    grayImage = np.zeros(img.shape)
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])

    R = (R *.299)
    G = (G *.587)
    B = (B *.114)

    Avg = (R+G+B)
    grayImage = img.copy()

    for i in range(3):
        grayImage[:,:,i] = Avg
        
    return grayImage  
    
a = datetime.datetime.now()
face = imageio.imread('lion.pgm')

# image = mpimg.imread('lion.png')   
# face = rgb_to_gray(image)  

# face = Image.open('lion.png')
# face = face.convert('L')
# print("TYPE imgray: ", type(face), face)

# print("TYPE face: ", type(face), face)

print(face.shape)

convx = array([[1/16, 2/16, 1/16],
               [2/16, 4/16, 2/16],
               [1/16, 2/16, 1/16]])
l = face.shape[0]
b = face.shape[1]
padded = np.zeros((l+2, b+2))
for i in range(0, l):
    for j in range(0, b):
        padded[i+1][j+1] = face[i][j]


res = np.zeros((l, b), dtype='uint8')
i = None
j = None

for i in range(1, l+1):
    for j in range(1, b+1):
        res[i-1][j-1] = (convx[0][0]*padded[i-1][j-1] + convx[0][1]*padded[i-1][j]+convx[0][2]*padded[i-1][j+1] +
                         convx[1][0]*padded[i][j-1]+convx[1][1]*padded[i][j] + convx[1][2]*padded[i][j+1] +
                         convx[2][0]*padded[i+1][j-1] + convx[2][1]*padded[i+1][j] + convx[2][2]*padded[i+1][j+1])


img = Image.fromarray(res)
img.save('GBS.png')
b = datetime.datetime.now()
print(b-a)

# print("TYPE img: ", type(img), img)

# img = Image.open('lion.png')
# imgGray = img.convert('L')
# print("TYPE imgray: ", type(imgGray), imgGray)

# print("TYPE res: ", type(res), res)
