#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:48:22 2018

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
#from ortools.constraint_solver import pywrapcp
import glob

#/device:GPU:0
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


from imageai.Detection import VideoObjectDetection
import os

#with tf.device("/device:GPU:0"):

execution_path = os.getcwd()


detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "traffic.mp4"),output_file_path=os.path.join(execution_path, "traffic_detected"), frames_per_second=20, log_progress=True)    
print(video_path)
    
    