'''

1. Gaussian 
2. OTSU
3. SobelEdge
4. Hough Transform

'''

import numpy as np
from numpy import array
from PIL import Image
import datetime
import imageio

MAX_IMAGESIZE = 4000
MAX_BRIGHTNESS = 255
GRAYLEVEL = 256
MAX_FILENAME = 256
MAX_BUFFERSIZE = 256


def otsu_th():
        
    face = imageio.imread('Flow/output1.pgm')
    image2 = face
    print(face.shape)

    image1 = face

    y_size1 = face.shape[0]
    x_size1 = face.shape[1]
    hist = [0]*256
    prob = [0.0]*256
    myu = [0.0]*256
    omega = [0.0]*256
    sigma = [0.0]*256
    
    print("Otsu's binarization process starts now.\n")
    # /* Histogram generation */
    for y in range(0, y_size1):
        for x in range(0, x_size1):
            hist[image1[y][x]] += 1

    # /* calculation of probability density */
    for i in range(0, GRAYLEVEL):
        prob[i] = float(hist[i]) / (x_size1 * y_size1)
    for i in range(0, 256):
        print("Serial: " + str(prob[i]))
    # /* omega & myu generation */
    omega[0] = prob[0]
    myu[0] = 0.0  # /* 0.0 times prob[0] equals zero */
    for i in range(1, GRAYLEVEL):
        omega[i] = omega[i-1] + prob[i]
        myu[i] = myu[i-1] + i*prob[i]

    '''/* sigma maximization
     sigma stands for inter-class variance 
     and determines optimal threshold value */'''
    threshold = 0
    max_sigma = 0.0
    for i in range(0, GRAYLEVEL-1):
        if (omega[i] != 0.0 and omega[i] != 1.0):
            sigma[i] = ((myu[GRAYLEVEL-1]*omega[i] - myu[i])
                        ** 2) / (omega[i]*(1.0 - omega[i]))
        else:
            sigma[i] = 0.0
        if (sigma[i] > max_sigma):
            max_sigma = sigma[i]
            threshold = i

    print("\nthreshold value = " + str(threshold))

    # /* binarization output into image2 */
    x_size2 = x_size1
    y_size2 = y_size1
    for y in range(0, y_size2):
        for x in range(0, x_size2):
            if (image1[y][x] > threshold):
                image2[y][x] = MAX_BRIGHTNESS
            else:
                image2[y][x] = 0
    print("End")

    return image2


def gaussianBlur():
    a = datetime.datetime.now()
    face = imageio.imread('lion.pgm')

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
    img.save('Flow/output1.pgm')
    b = datetime.datetime.now()
    print(b-a)

def OTSU():
    a = datetime.datetime.now()
    image2 = otsu_th()
    b = datetime.datetime.now()
    print("Time: "+str(b-a))
    img = Image.fromarray(image2)
    img.save('Flow/output2.pgm')
    # img.show()

def sobelEdge():

    a = datetime.datetime.now()
    face = imageio.imread('Flow/output2.pgm')

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
    img.save('Flow/output3.png')
    # img.show()

    b = datetime.datetime.now()
    print(b-a)


'''Main execution'''

gaussianBlur()

OTSU()

sobelEdge()