# -*- coding: utf-8 -*-
"""
Spyder Editor
Editor : logicalsteps
"""

from flask import Flask, render_template, request
from flask import redirect, url_for
from werkzeug import secure_filename
from imageai.Detection import ObjectDetection
import os
import pandas as pd
import numpy as np
from keras import backend as K      
import random
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as ks
from keras import metrics
import glob
from PIL import Image
import pytesseract
import sklearn
import cv2

app = Flask(__name__, root_path = os.getcwd())

def remove_detected_image():
    detected_image_path = './detection/'
    for file in os.listdir(detected_image_path):
        if (file.endswith(".jpg") or file.endswith(".jpeg")):
            os.remove(file)	

def remove_uploaded_image():
    uploaded_image_path = os.getcwd()
    for file in os.listdir(uploaded_image_path):
        if (file.endswith(".jpg") or file.endswith(".jpeg")):
            os.remove(file)	

@app.route('/')
def Home_renderer():
    return render_template("upload.html")

@app.route('/indexImageDetection',methods = ['GET', 'POST'])
def ImageDetection_renderer():
   if request.method == 'POST':
      f = request.files['file']
      filename = 'ImageDetectionUpload.jpg'
      f.save(secure_filename(filename))
   if request.method == 'POST':
      execution_path = os.getcwd()
      detector = ObjectDetection()
      detector.setModelTypeAsRetinaNet()
      detector.setModelPath( os.path.join("resnet50_coco_best_v2.0.1.h5"))
      detector.loadModel()
      input_name = "ImageDetectionUpload.jpg"
      ImageName = "./detection/ImageDetection.jpg"
      detections = detector.detectObjectsFromImage(input_image=input_name, output_image_path=ImageName, minimum_percentage_probability=30)
      for eachObject in detections:
         print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
      K.clear_session()
      return render_template('ImageDetectionDisplay.html',imagedetected='../detection/ImageDetection.jpg',imagesource = '../ImageDetectionUpload.jpg')
   
@app.route('/indexOCR',methods = ['GET', 'POST'])
def Ocr_renderer():
   if request.method == 'POST':
      f = request.files['file']
      filename = 'OCRUpload.png'
      f.save(secure_filename(filename))
      text = pytesseract.image_to_string(Image.open("OCRUpload.png"))
      return text


@app.route('/indexFaceDetection',methods = ['GET', 'POST'])
def FaceDetection_renderer():
   if request.method == 'POST':
      f = request.files['file']
      filename = 'FaceDetectionUpload.jpg'
      f.save(secure_filename(filename))
      face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
      eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

   img = cv2.imread('FaceDetectionUpload.jpg')
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray, 1.3, 5)
   for (x,y,w,h) in faces:
     img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
     roi_gray = gray[y:y+h, x:x+w]
     roi_color = img[y:y+h, x:x+w]
     eyes = eye_cascade.detectMultiScale(roi_gray)
     for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
     cv2.imwrite("./detection/FaceDetection.jpg",img)
   
   return render_template('FaceDetectionDisplay.html',imagedetected='../detection/FaceDetection.jpg',imagesource = '../FaceDetectionUpload.jpg')


if __name__ == '__main__':
   app.run(debug = True)



