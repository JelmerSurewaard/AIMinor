from flask import Flask, flash, request, redirect, url_for, render_template
import os

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

from PIL import Image

from werkzeug.utils import secure_filename

app = Flask(__name__, )

UPLOAD_FOLDER = 'static/uploads/'

CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/check", methods=["GET"])
def check():
    return render_template('check.html')

@app.route('/upload', methods=['GET'])
def upload():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
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

        model = keras.models.load_model('model_V2')

        # Rescale image to 48 x 48, then grayscale image
        img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)).convert('L')
        res = img.resize((48, 48), Image.ANTIALIAS)
        res.save(os.path.join(app.config['UPLOAD_FOLDER'], "temp.jpg"))

        image = tf.keras.preprocessing.image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], "temp.jpg"))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch.

        prediction_own = model.predict(input_arr)
        df_own = pd.DataFrame(prediction_own, columns=CLASS_LABELS)

        val = df_own.max(axis=1)
        val2 = df_own.idxmax(axis=1)

        # print(val.values[0])
        # print(val2.values[0])

        tag = val2.values[0]
        precentage = val.values[0]

        # Remove the files after use:
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], "temp.jpg"))

        # print('upload_image filename: ' + filename)
        # flash('Image successfully uploaded and displayed below')
        return render_template('check.html', percentage=precentage, tag=tag)
    else:
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=50000)