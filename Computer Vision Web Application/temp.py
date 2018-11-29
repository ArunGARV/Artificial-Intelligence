# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import flask
import sklearn

import tensorflow as tf
import keras
import cv2
import numpy as np
import pandas as pd
import imageai
from imageai import Prediction
#from imageai.Prediction import ImagePrediction
from imageai.Detection import ObjectDetection
import os
execution_path = os.getcwd()
import glob
import random


detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
#detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.setModelPath( os.path.join("resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
#for file in os.listdir(execution_path):
#if (file.endswith(".jpg") or file.endswith(".jpeg")):
      
list_of_files =  glob.glob('*.jpg') + glob.glob('*.jpeg') + glob.glob('*.png')
input_name = max(list_of_files, key=os.path.getctime)
#ImageName = "./detection/Detection"+str(random.randint(1,10000))+".jpg"
ImageName = "./detection/Detection"+".jpg"
#detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path,input_name), output_image_path = ImageName, minimum_percentage_probability=70)
detections = detector.detectObjectsFromImage(input_image=input_name, output_image_path=ImageName, minimum_percentage_probability=30)
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
