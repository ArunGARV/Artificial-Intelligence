#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:48:54 2018

@author: logicalsteps

"""


from flask import Flask, render_template, request
from flask import Flask, redirect, url_for
from werkzeug import secure_filename
from imageai.Detection import ObjectDetection, VideoObjectDetection
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



import cv2

execution_path = os.getcwd()


camera = cv2.VideoCapture(0)

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
video_path = detector.detectObjectsFromVideo(camera_input=camera,output_file_path=os.path.join(execution_path, "camera_detected_video"), frames_per_second=20, log_progress=True, minimum_percentage_probability=40)

'''
import datetime
from datetime import date
now = datetime.date.today()
d1 = date(2018, 11, 26)
delta = d1 - now
print (delta.days)
'''