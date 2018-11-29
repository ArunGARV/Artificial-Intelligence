import os
 
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import numpy as np
import argparse
import imutils
import cv2
import time
import uuid
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'detection'

@app.route("/")
def template_test():
    return render_template('templateDemo.html', label='', imagesource='../detection/FaceDetection.jpg')



if __name__ == "__main__":
    app.debug=False
    app.run(host='0.0.0.0', port=5000)
