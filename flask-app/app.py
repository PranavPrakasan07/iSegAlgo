from flask import Flask, render_template, redirect, flash, request, url_for
from werkzeug.utils import secure_filename
import urllib.request
import os
from PIL import Image, ImageOps
import time

import numpy as np
from numpy import array
from PIL import Image
import datetime
import imageio

app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if app.config["DEBUG"]:
    @app.after_request
    def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'pgm'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_pmg(filename):
    img = Image.open(os.path.join("./", filename))
    gray_image = ImageOps.grayscale(img)
    gray_image.save(os.path.join("./", "gscale.pgm"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join("./", filename))
        flash('Image successfully uploaded')
        convert_to_pmg(filename)
        return render_template('home.html', enable = True)

    else:
        flash('Invalid file type: Not allowed')
        return redirect(request.url)

@app.route('/process', methods=['GET'])
def process_upload():
    t_gb_s = GaussianBlurSerial()
    t_otsu_s = OtsuSerial()
    t_sobel_s = SobelSerial()

    gbs = ["GBS.png", t_gb_s]
    otsus = ["OTSUS.png", t_otsu_s]
    seds = ["SEDS.png", t_sobel_s]
    return render_template('home.html', gbs = gbs, otsus = otsus, seds = seds)


    
    # t_gb_s = sample.hello()  # execute the function GB serial
    # t_gb_p = sample.hello()  # execute the function GB parallel
    # t_otsu_s = sample.hello()  # execute the function OTSU serial
    # t_otsu_p = sample.hello()  # execute the function OTSU parallel
    # t_sobel_s = sample.hello()  # execute the function SOBEL serial
    # t_sobel_p = sample.hello()  # execute the function SOBEL parallel
    # gbs = ["GBS.png", t_gb_s]
    # gbp = ["GBP.png", t_gb_p]
    # otsus = ["OTSUS.png", t_otsu_s]
    # otsup = ["OTSUP.png", t_otsu_p]
    # seds = ["SOBELS.png", t_sobel_s]
    # sedp = ["SOBELP.png", t_sobel_p]
    # return render_template('home.html', gbs=gbs, gbp=gbp, otsus=otsus, otsup=otsup, seds=seds, sedp=sedp)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='result/' + filename), code=301)



def GaussianBlurSerial():
    a = datetime.datetime.now()
    face = imageio.imread('gscale.pgm')

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
    img.save('./static/result/GBS.png')
    b = datetime.datetime.now()
    return (b-a).total_seconds()


def OtsuSerial():
    MAX_IMAGESIZE = 4000
    MAX_BRIGHTNESS = 255
    GRAYLEVEL = 256
    MAX_FILENAME = 256
    MAX_BUFFERSIZE = 256

    face = imageio.imread('gscale.pgm')
    image1 = face
    image2 = face
    print(face.shape)
    y_size1 = face.shape[0]
    x_size1 = face.shape[1]
    hist = [0]*256
    prob = [0.0]*256
    myu = [0.0]*256
    omega = [0.0]*256
    sigma = [0.0]*256

    def otsu_th():
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

        # print("\nthreshold value = " + str(threshold))

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


    a = datetime.datetime.now()
    otsu_th()
    b = datetime.datetime.now()
    img = Image.fromarray(image2)
    img.save('./static/result/OTSUS.png')
    return (b-a).total_seconds()


def SobelSerial():
    a = datetime.datetime.now()
    face = imageio.imread('gscale.pgm')

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
    img.save('./static/result/SEDS.png')
    b = datetime.datetime.now()
    return (b-a).total_seconds()