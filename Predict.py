import numpy as np
import cv2 as cv
from cv2 import cv2
from sklearn.feature_extraction import image
from whenet import WHENet
from utils import draw_axis
import os
import argparse
from PIL import Image
import time
import pickle
import tensorflow as tf
from face import extractFaces, detectPoints




import tensorflow as tf
graph = tf.get_default_graph()
classify_model = pickle.load(open("./model/model.pkl","rb"))

def process_detection( model, img ):
    # face aligment
    face = detectPoints(img)
    bbox = extractFaces(face)
    if bbox is None:
        return img
    y_min, y_max, x_min, x_max = bbox

    img_rgb = img[int(y_min):int(y_max), int(x_min):int(x_max)]
    #img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))
    img_rgb = np.expand_dims(img_rgb, axis=0)
    #cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,0,0), 2)
    with graph.as_default():
        yaw, pitch, roll = model.get_angle(img_rgb)
    yaw, pitch, roll = np.squeeze([yaw, pitch, roll])
    #draw_axis(img, yaw, pitch, roll, tdx=(x_min+x_max)/2, tdy=(y_min+y_max)/2, size = abs(x_max-x_min)//2 )

    pred = classify_model.predict([[pitch, yaw]])
    cv2.putText(img, "text: {}".format(pred[0]), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #cv2.putText(image, "yaw: {}".format(np.round(yaw)), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
    #cv2.putText(image, "pitch: {}".format(np.round(pitch)), (int(x_min), int(y_min) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
    #cv2.putText(image, "roll: {}".format(np.round(roll)), (int(x_min), int(y_min)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
    return img
