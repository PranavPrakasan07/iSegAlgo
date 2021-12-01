from __future__ import division
import cv2
import numpy as np
from PIL import Image
import datetime
import pymp

pymp.config.nested = True

a = datetime.datetime.now()

original_img_path = "cans.png"
original_image = cv2.imread(original_img_path, 0)
#gray_image = cv2.imread('Sample_Input.jpg',0)
img = Image.fromarray(original_image)
img.save("HTC_Original_P.png")

output = original_image.copy()

# Gaussian Blurring of Gray Image
blur_image = cv2.GaussianBlur(original_image, (3, 3), 0)
img = Image.fromarray(blur_image)
img.save("HTC_Blur_P.png")

# Using OpenCV Canny Edge detector to detect edges
edged_image = cv2.Canny(blur_image, 75, 150)
img = Image.fromarray(edged_image)
img.save("HTC_Edged_P.png")

height, width = edged_image.shape
radii = 100

acc_array = np.zeros(((height, width, radii)))

filter3D = np.zeros((30, 30, radii))
filter3D[:, :, :] = 1


def fill_acc_array(x0, y0, radius):
    x = radius
    y = 0
    decision = 1-x

    while(y < x):
        if(x + x0 < height and y + y0 < width):
            acc_array[x + x0, y + y0, radius] += 1  # Octant 1
        if(y + x0 < height and x + y0 < width):
            acc_array[y + x0, x + y0, radius] += 1  # Octant 2
        if(-x + x0 < height and y + y0 < width):
            acc_array[-x + x0, y + y0, radius] += 1  # Octant 4
        if(-y + x0 < height and x + y0 < width):
            acc_array[-y + x0, x + y0, radius] += 1  # Octant 3
        if(-x + x0 < height and -y + y0 < width):
            acc_array[-x + x0, -y + y0, radius] += 1  # Octant 5
        if(-y + x0 < height and -x + y0 < width):
            acc_array[-y + x0, -x + y0, radius] += 1  # Octant 6
        if(x + x0 < height and -y + y0 < width):
            acc_array[x + x0, -y + y0, radius] += 1  # Octant 8
        if(y + x0 < height and -x + y0 < width):
            acc_array[y + x0, -x + y0, radius] += 1  # Octant 7
        y += 1
        if(decision <= 0):
            decision += 2 * y + 1
        else:
            x = x-1
            decision += 2 * (y - x) + 1


edges = np.where(edged_image == 255)

with pymp.Parallel(10) as p1:
    for i in p1.range(0, len(edges[0])):
        x = edges[0][i]
        y = edges[1][i]

        with pymp.Parallel(10) as p2:
            for radius in p2.range(0, 55):
                fill_acc_array(x, y, radius)


i = 0
j = 0
while(i < height-30):
    while(j < width-30):
        filter3D = acc_array[i:i+30, j:j+30, :]*filter3D
        max_pt = np.where(filter3D == filter3D.max())
        a = max_pt[0]
        b = max_pt[1]
        c = max_pt[2]
        b = b+j
        a = a+i
        if(filter3D.max() > 45):
            cv2.circle(output, (b, a), c, (0, 255, 0), 2)
        j = j+30
        filter3D[:, :, :] = 1
    j = 0
    i = i+30

print("reached here")
img = Image.fromarray(output)
img.save("HTCP.png")

b = datetime.datetime.now()
print(b-a)
