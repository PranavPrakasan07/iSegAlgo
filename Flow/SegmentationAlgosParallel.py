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

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
from scipy.signal import fftconvolve
from scipy.ndimage import filters
import pymp

pymp.config.nested = True

_MIN_RADIUS = 15
_MAX_RADIUS = 75
_RADIUS_STEP = 5
_ANNULUS_WIDTH = 5
_EDGE_THRESHOLD = 0.005
_NEG_INTERIOR_WEIGHT = 1.1

MAX_IMAGESIZE = 4000
MAX_BRIGHTNESS = 255
GRAYLEVEL = 256
MAX_FILENAME = 256
MAX_BUFFERSIZE = 256

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
    with pymp.Parallel(2) as p1:
        with pymp.Parallel(2) as p2:
            for y in p1.range(0, y_size1):
                for x in p2.range(0, x_size1):
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
    with pymp.Parallel(2) as p1:
        with pymp.Parallel(2) as p2:
            for y in p1.range(0, y_size2):
                for x in p2.range(0, x_size2):
                    if (image1[y][x] > threshold):
                        image2[y][x] = MAX_BRIGHTNESS
                    else:
                        image2[y][x] = 0
    print("End")

    return image2


def gaussianBlur():
    a = datetime.datetime.now()
    
    # image = imageio.imread("Flow/lion.pgm")   

    # grayImage = rgb_to_gray(image)  
    # img = Image.fromarray(grayImage)
    # img.save('Flow/input.pgm')

    face = imageio.imread('Flow/cans.pgm')

    print(face.shape)

    convx = array([[1/16, 2/16, 1/16],
                [2/16, 4/16, 2/16],
                [1/16, 2/16, 1/16]])
    l = face.shape[0]
    b = face.shape[1]
    padded = np.zeros((l+2, b+2))
    with pymp.Parallel(2) as p1:
        with pymp.Parallel(2) as p2:
            for i in p1.range(0, l):
                for j in p2.range(0, b):
                    padded[i+1][j+1] = face[i][j]


    res = np.zeros((l, b), dtype='uint8')
    i = None
    j = None

    with pymp.Parallel(2) as p1:
        with pymp.Parallel(2) as p2:
            for i in p1.range(1, l+1):
                for j in p2.range(1, b+1):
                    res[i-1][j-1] = (convx[0][0]*padded[i-1][j-1] + convx[0][1]*padded[i-1][j]+convx[0][2]*padded[i-1][j+1] +
                                    convx[1][0]*padded[i][j-1]+convx[1][1]*padded[i][j] + convx[1][2]*padded[i][j+1] +
                                    convx[2][0]*padded[i+1][j-1] + convx[2][1]*padded[i+1][j] + convx[2][2]*padded[i+1][j+1])


    img = Image.fromarray(res)
    img.save('Flow/output1.pgm')
    img.save('Flow/output1.png') 

    b = datetime.datetime.now()
    print(b-a)

def OTSU():
    a = datetime.datetime.now()
    image2 = otsu_th()
    b = datetime.datetime.now()
    print("Time: "+str(b-a))
    img = Image.fromarray(image2)
    img.save('Flow/output2.pgm')
    img.save('Flow/output2.png')

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

    with pymp.Parallel(2) as p1:
        with pymp.Parallel(2) as p2:
            for i in p1.range(1, l+1):
                for j in p2.range(1, b+1):
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

    with pymp.Parallel(2) as p1:
        with pymp.Parallel(2) as p2:
            for i in p1.range(1, l+1):
                for j in p2.range(1, b+1):
                    resy[i-1][j-1] = (convy[0][0]*padded[i-1][j-1] + convy[0][1]*padded[i-1][j]+convy[0][2]*padded[i-1][j+1] +
                                    convy[1][0]*padded[i][j-1]+convy[1][1]*padded[i][j] + convy[1][2]*padded[i][j+1] +
                                    convy[2][0]*padded[i+1][j-1] + convy[2][1]*padded[i+1][j] + convy[2][2]*padded[i+1][j+1])

                    resy[i-1][j-1] = (resy[i-1][j-1]**2)

    res2 = np.zeros((l,b)).astype(np.uint8)

    with pymp.Parallel(2) as p1:
        with pymp.Parallel(2) as p2:
            for i in p1.range(0, l):
                for j in p2.range(0, b):
                    res2[i][j] = int((res[i-1][j-1]+int(resy[i-1][j-1])))
                    if res2[i][j] > 15:
                        res2[i][j] = 255


    img = Image.fromarray(res2.astype(np.uint8))
    img.save('Flow/output3.png')
    # img.show()

    b = datetime.datetime.now()
    print(b-a)



def _detectEdges(image, threshold):
    """
    Sobel edge detection on the image
    """
    # sobel filter in x and y direction
    image = filters.sobel(image, 0)**2 + filters.sobel(image, 1)**2
    image -= image.min()

    # make binary image
    image = image > image.max()*threshold
    image.dtype = np.int8

    return image


def _makeAnnulusKernel(outer_radius, annulus_width):
    """
    Create an annulus with the given inner and outer radii

    Ex. inner_radius = 4, outer_radius = 6 will give you:


    """
    grids = np.mgrid[-outer_radius:outer_radius +
                     1, -outer_radius:outer_radius+1]

    # [j][i] = r^2
    kernel_template = grids[0]**2 + grids[1]**2

    # get boolean value for inclusion in the circle
    outer_circle = kernel_template <= outer_radius**2
    inner_circle = kernel_template < (outer_radius - annulus_width)**2

    # back to integers
    outer_circle.dtype = inner_circle.dtype = np.int8
    inner_circle = inner_circle*_NEG_INTERIOR_WEIGHT
    annulus = outer_circle - inner_circle
    return annulus


def _detectCircles(image, radii, annulus_width):
    """
    Perfrom a FFT Convolution over all the radii with the given annulus width.
    Smaller annulus width = more precise
    """
    acc = np.zeros((radii.size, image.shape[0], image.shape[1]))

    for i, r in enumerate(radii):
        C = _makeAnnulusKernel(r, annulus_width)
        acc[i, :, :] = fftconvolve(image, C, 'same')

    return acc


def _iterativeDetectCircles(edges, image):
    """
    TODO: finish this

    The idea:
    Start with an annulus with a large radius.
    Split this into 2 annuli of equal area. The annulus width will be different
    Find the annulus with a higher signal in the image
    Repeat the process for the higher signal, 
    until a minumum annulus width is reached (preferrably 1)

    Initialize:
    large radius = min(image.shape)/2 ... actually maybe a little less
    min radius is some small number, not too small to avoid high match to noise

    TODO: the signals for each annulus have to be comparable
             solution? -> normalize to the area of the annulus
    """

    return


def _displayResults(image, edges, center, radius, output=None):
    """
    Display the accumulator for the radius with the highest votes.
    Draw the radius on the image and display the result.
    """

    # display accumulator image
    plt.gray()
    fig = plt.figure(1)
    fig.clf()
    subplots = []
    subplots.append(fig.add_subplot(1, 2, 1))
    plt.imshow(edges)
    plt.title('Edge image')

    # display original image
    subplots.append(fig.add_subplot(1, 2, 2))
    plt.imshow(image)
    plt.title('Center: %s, Radius: %d' % (str(center), radius))
    # draw the detected circle
    blob_circ = plt_patches.Circle(center, radius, fill=False, ec='red')
    plt.gca().add_patch(blob_circ)

    #   Fix axis distortion:
    plt.axis('image')

    if output:
        plt.savefig(output)

    plt.draw()
    plt.show()

    return


def _topNCircles(acc, radii, n):

    maxima = []
    max_positions = []
    max_signal = 0
    circle_x = circle_y = radius = 0

    for i, r in enumerate(radii):
        max_positions.append(np.unravel_index(acc[i].argmax(), acc[i].shape))
        maxima.append(acc[i].max())
        # use the radius to normalize
        signal = maxima[i]/np.sqrt(float(r))

        if signal > max_signal:
            max_signal = signal
            (circle_y, circle_x) = max_positions[i]
            radius = r
        print("Maximum signal for radius %d: %d %s, normal signal: %f" %
              (r, maxima[i], max_positions[i], signal))

    # Identify maximum. Note: the values come back as index, row, column
#    max_index, circle_y, circle_x = np.unravel_index(acc.argmax(), acc.shape)

    return (circle_x, circle_y), radius  # radii[max_index]


def DetectCircleFromFile(filename, show_result=False):

    image = plt.imread(filename)
    center, radius = DetectCircle(image, True, show_result)
    return center, radius


def DetectCircle(image, preprocess=False, show_result=False):

    if preprocess:
        if image.ndim > 2:
            image = np.mean(image, axis=2)
        print("Image size: ", image.shape)

        # noise reduction
        image = filters.gaussian_filter(image, 2)

        # edges and density
        edges = _detectEdges(image, _EDGE_THRESHOLD)
        edge_list = np.array(edges.nonzero())
        density = float(edge_list[0].size)/edges.size
        print("Signal density:", density)
        if density > 0.25:
            print("High density, consider more preprocessing")

    # create kernels and detect circle
    radii = np.arange(_MIN_RADIUS, _MAX_RADIUS, _RADIUS_STEP)
    acc = _detectCircles(edges, radii, _ANNULUS_WIDTH)
    center, radius = _topNCircles(acc, radii, 1)
    print("Circle detected at ", center, radius)

    if show_result:
        _displayResults(image, edges, center, radius, 'Flow/output4.png')

    return center, radius


def _run(filename):
    return DetectCircleFromFile(filename)


def houghTranform():
    DetectCircleFromFile('Flow/output3.png', True)


'''Main execution'''

gaussianBlur()

OTSU()

sobelEdge()

houghTranform()