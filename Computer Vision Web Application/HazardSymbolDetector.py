#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:32:28 2018

@author: logicalsteps
"""


from flask import Flask, render_template, request
from flask import Flask, redirect, url_for
from werkzeug import secure_filename
from imageai.Detection import ObjectDetection
import os
import pandas as pd
import numpy as np
from keras import backend as K
      
import random
import bokeh
from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.tools import HoverTool
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import keras as ks
from keras import metrics
from ortools.constraint_solver import pywrapcp
import glob


app = Flask(__name__)
#app.config[‘UPLOAD_FOLDER’]
#app.config['MAX_CONTENT_PATH']

def remove_detected_image():
    detected_image_path = './detection/'
    for file in os.listdir(detected_image_path):
        if (file.endswith(".jpg") or file.endswith(".jpeg")):
            os.remove(file)	

def remove_uploaded_image():
    #os.chdir('/home/logicalsteps/Computer Vision/FlaskDemo1/')
    uploaded_image_path = os.getcwd()
    for file in os.listdir(uploaded_image_path):
        if (file.endswith(".jpg") or file.endswith(".jpeg")):
            os.remove(file)	


@app.route('/upload')
def html_renderer():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return render_template('upload1.html')
		

@app.route('/detection', methods = ['GET', 'POST'])
def detect():
   if request.method == 'POST':
      #remove_detected_image()
      execution_path = os.getcwd()
      detector = ObjectDetection()
      detector.setModelTypeAsRetinaNet()
      detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
      detector.loadModel()
      #for file in os.listdir(execution_path):
         #if (file.endswith(".jpg") or file.endswith(".jpeg")):
      
      list_of_files = glob.glob('./*.jpg')
      input_image = max(list_of_files, key=os.path.getctime)
      #print (latest_file) 
      #input_image = os.path.join(execution_path , file)
      ImageName = "./detection/Detection"+str(random.randint(1,10000))+".jpg"
      detections = detector.detectObjectsFromImage(input_image, output_image_path = ImageName)
      #output_image_path=os.path.join(execution_path , "imagenew.jpg"))
      for eachObject in detections:
         print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
      #remove_uploaded_image()
      #return 'Succesful detection'
      K.clear_session()
      return redirect(url_for('html_renderer'))
   
    
if __name__ == '__main__':
   app.run(debug = True)





