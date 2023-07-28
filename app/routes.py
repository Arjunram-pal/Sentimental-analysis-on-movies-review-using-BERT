

import os
import csv
from flask import Flask, flash, request, redirect, send_file, render_template
import tensorflow as tf
import pandas as pd
import numpy as np
import ktrain
from ktrain import text
from werkzeug.utils import secure_filename

tf.random.set_seed(42)



UPLOAD_FOLDER = 'uploads/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pretrained model and vectorizer

predictor_load=ktrain.load_predictor('C:/Users/Lenovo/Downloads/bert_model')


@app.route("/")
@app.route("/index")
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = [message]
    my_prediction = predictor_load.predict(data)
    print(my_prediction)
    return render_template('result.html', prediction=my_prediction)

