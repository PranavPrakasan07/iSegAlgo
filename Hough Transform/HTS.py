import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
from scipy.signal import fftconvolve
from scipy.ndimage import filters

_MIN_RADIUS = 15
_MAX_RADIUS = 75
_RADIUS_STEP = 5
_ANNULUS_WIDTH = 5
_EDGE_THRESHOLD = 0.005
_NEG_INTERIOR_WEIGHT = 1.1


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
        _displayResults(image, edges, center, radius)

    return center, radius


def _run(filename):
    return DetectCircleFromFile(filename)


def _test():
    #    ## TODO: test cases, especially for iterative approach
    #    radii = np.arange(_MIN_RADIUS, _MAX_RADIUS, _RADIUS_STEP)
    #    image, edges = _initialize('mri.png')
    #    acc = _detectCircles(edges, radii, _ANNULUS_WIDTH)
    #    center, radius = _topNCircles(acc, radii, 1)
    #    print "Circle detected at ", center, radius
    #    _displayResults(image, edges, acc, radii, "mri-result.png")
    #    return
    DetectCircleFromFile('Hough Transform/cans.pgm', True)
#    DetectCircleFromVideo("/home/vinnie/workspace/Intelligent-Artifacts/AMID/data/MRI_SAX_AVI/321/(MAIN)04052012-173005.avi")


if __name__ == "__main__":
    if (len(sys.argv) > 1):
        _run(sys.argv[1])
    else:
        _test()
