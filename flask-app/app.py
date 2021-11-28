from flask import Flask, render_template, redirect, flash, request, url_for
from werkzeug.utils import secure_filename
import urllib.request
import os
from PIL import Image, ImageOps

from algorithms import sample

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/result/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','pgm'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_pmg(filename):
    img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    gray_image = ImageOps.grayscale(img)
    gray_image.save(os.path.join(app.config['UPLOAD_FOLDER'], "gscale.pgm"))

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
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded')
        convert_to_pmg(filename)
 
        t_gb_s = sample.hello() #execute the function GB serial
        t_gb_p = sample.hello() #execute the function GB parallel
        t_otsu_s = sample.hello() #execute the function OTSU serial
        t_otsu_p = sample.hello() #execute the function OTSU parallel
        t_sobel_s = sample.hello() #execute the function SOBEL serial
        t_sobel_p = sample.hello() #execute the function SOBEL parallel

        gbs = ["GBS.png", t_gb_s]
        gbp = ["GBP.png", t_gb_p]
        
        otsus = ["OTSUS.png", t_otsu_s]
        otsup = ["OTSUP.png", t_otsu_p]

        seds = ["SOBELS.png", t_sobel_s]
        sedp = ["SOBELP.png", t_sobel_p]

        return render_template('home.html', gbs = gbs, gbp = gbp, otsus = otsus, otsup = otsup, seds = seds, sedp = sedp)
    else:
        flash('Invalid file type: Not allowed')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='result/' + filename), code=301)