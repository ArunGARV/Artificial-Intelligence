# -*- coding: utf-8 -*-
"""
Spyder Editor
Editor : logicalsteps
"""

from flask import Flask, render_template, request
#from flask import redirect, url_for
from werkzeug import secure_filename
from imageai.Detection import ObjectDetection
import os
#import pandas as pd
import numpy as np
from keras import backend as K      
#import random
#from datetime import datetime
#import matplotlib.pyplot as plt
import tensorflow as tf
#import keras as ks
#from keras import metrics
#import glob
from PIL import Image
import pytesseract
#import sklearn
import cv2
#import sys
os.getcwd()
from utils import label_map_util
from utils import visualization_utils as vis_util

app = Flask(__name__, root_path = os.getcwd())

###########################Preemptive measures to remove pre-existing images####################
################################################################################################
################################################################################################
################################################################################################

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
'''
@app.route('/')
def Main_renderer():
   return render_template("upload.html")
'''

###########################Hazchem Symbol, UnNo, Hazchem Code detection#########################
################################################################################################
################################################################################################
################################################################################################
'''
@app.route('/indexHazchemImageDetection',methods = ['GET', 'POST'])
def HazchemDetection_renderer():
   if request.method == 'POST':
      f = request.files['file']
      IMAGE_NAME = 'HazchemDetectionUpload.jpg'
      f.save(secure_filename(IMAGE_NAME))
      MODEL_NAME = 'inference_graph'
      #IMAGE_NAME = 'test3.jpg'

      CWD_PATH = os.getcwd()
      PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
      PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
      PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
      NUM_CLASSES = 3
      label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
      categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
      category_index = label_map_util.create_category_index(categories)

      detection_graph = tf.Graph()
      with detection_graph.as_default():
         od_graph_def = tf.GraphDef()
         with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

         sess = tf.Session(graph=detection_graph)

      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
      detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      image = cv2.imread(PATH_TO_IMAGE)
      image_expanded = np.expand_dims(image, axis=0)

      (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_expanded})

      vis_util.visualize_boxes_and_labels_on_image_array(image,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=8,min_score_thresh=0.80)

      copy=tf.image.draw_bounding_boxes(image,boxes,name=None)

      height, width = image.shape[:2]

      ymin = boxes[0][0][0]*height
      xmin = boxes[0][0][1]*width
      ymax = boxes[0][0][2]*height
      xmax = boxes[0][0][3]*width

      cv2.imwrite("./detection/HazchemDetection.jpg",image)
      return render_template("HazchemDetectionDisplay.html")

'''
###########################Image detection using Resnet model###################################
################################################################################################
################################################################################################
################################################################################################

@app.route('/indexImageDetection',methods = ['GET', 'POST'])
def ImageDetection_renderer():
   if request.method == 'POST':
      f = request.files['file']
      filename = 'ImageDetectionUpload.jpg'
      f.save(secure_filename(filename))
   if request.method == 'POST':
      #execution_path = os.getcwd()
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


###########################OPtical Character Recognition using pytesseract######################
################################################################################################
################################################################################################
################################################################################################
   
@app.route('/indexOCR',methods = ['GET', 'POST'])
def Ocr_renderer():
   if request.method == 'POST':
      f = request.files['file']
      filename = 'OCRUpload.png'
      f.save(secure_filename(filename))
      text = pytesseract.image_to_string(Image.open("OCRUpload.png"))
      return text

###########################Detecting Faces and Eyes using HaarCascade features##################
################################################################################################
################################################################################################
################################################################################################

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



