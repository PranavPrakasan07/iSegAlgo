import datetime
import cv2
import math
import numpy as np
from PIL import Image


def Hough_lines(img):

    height, width = img.shape[:2]
    accumulator = np.zeros(
        (180, int(math.sqrt(height ** 2 + width ** 2))), dtype=int)

    lines = np.array([[0, 0], [0, 0]])

    line_length = 10

    # look for every pixel
    for y in range(0, height):
        for x in range(0, width):
            # if pixel is black (possible part of a line)
            if img[y][x] < 20:
                line = []
                # try all angles
                for theta in range(0, 180):
                    p = int(x * math.cos(math.radians(theta)) +
                            y * math.sin(math.radians(theta)))
                    accumulator[theta][p] += 1
                    # Check if it looks like line and if it's not in a list
                    if (accumulator[theta][p] > line_length) and (p not in lines[:, 0]) and (theta not in lines[:, 1]):
                        lines = np.vstack((lines, np.array([p, theta])))

    # clean two first zeros
    lines = np.delete(lines, [0, 1], axis=0)

    return lines

# find all intersection


def line_intersection(p, theta, img):
    h, w = img.shape[:2]
    out = []
    theta = math.radians(theta)
    intersect = [int(round(p / math.sin(theta))), int(round((p - w * math.cos(theta)) / math.sin(theta))), int(round(p / math.cos(theta))),
                 int(round((p - h * math.sin(theta)) / math.cos(theta)))]
    if (intersect[0] > 0) and (intersect[0] < h):
        out.append((0, intersect[0]))
    if (intersect[1] > 0) and (intersect[1] < h):
        out.append((w, intersect[1]))

    if (intersect[2] > 0) and (intersect[2] < w):
        out.append((intersect[2], 0))
    if (intersect[3] > 0) and (intersect[3] < w):
        out.append((intersect[3], h))
    return out


def main():
    a = datetime.datetime.now()

    original_img_path = "cans.png"
    img = cv2.imread(original_img_path, 0)
    lines = Hough_lines(img)
    print("lines:", lines)

    for i in lines:
        points = line_intersection(i[0], i[1], img)
        print(points)
        cv2.line(img, points[0], points[1], [100])

    img = Image.fromarray(img)
    img.save('HTS.png')

    b = datetime.datetime.now()
    print(b-a)

if __name__ == "__main__":
    main()




