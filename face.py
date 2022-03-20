from skimage import exposure
import tensorflow as tf
import mediapipe as mp
from PIL import Image
import numpy as np  
import cv2 as cv
from cv2 import cv2
# Required definitions for MediaPipe face mesh (it is used for face alignment)


# Required definitions for face alignment (Mediapipe face Mesh)
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2, static_image_mode = True)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

# Required definitions for face detections(Mediapipe face Mesh)
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence = 0.5)



def detectPoints(img):
    left_eye, right_eye = [], []
    ih, iw, ic = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            for id,lm in enumerate(faceLms.landmark):
                x,y,z = int(lm.x*iw), int(lm.y*ih), lm.z
                # left eye coords
                if (id == 113) | (id == 243) | (id == 159) | (id == 145):
                    left_eye.append((x,y))
                # right eye coords
                elif (id == 263) | (id == 362) | (id == 386) | (id == 374):
                    right_eye.append((x,y))
            
        # left eye center (not iris)
        left_center_x = (left_eye[0][0] + left_eye[1][0] +left_eye[2][0] +left_eye[3][0]) // 4
        left_center_y = (left_eye[0][1] + left_eye[1][1] +left_eye[2][1] +left_eye[3][1]) // 4
        
        # rights eye center (not iris)
        right_center_x = (right_eye[0][0] + right_eye[1][0] +right_eye[2][0] +right_eye[3][0]) // 4
        right_center_y = (right_eye[0][1] + right_eye[1][1] +right_eye[2][1] +right_eye[3][1]) // 4
        
        # distance of two eyes(left-right)
        delta_x = right_center_x - left_center_x
        delta_y = right_center_y - left_center_y
        # angle of two eyes
        angle=np.arctan(delta_y/delta_x)
        angle = (angle * 180) / np.pi
                
        # Calculating a center point of the image
        # Integer division "//"" ensures that we receive whole numbers
        center = (iw // 2, ih // 2)
        # Defining a matrix M and calling cv2.getRotationMatrix2D method
        M = cv2.getRotationMatrix2D(center, (angle), 1.0)
        # Applying the rotation to our image using the cv2.warpAffine method
        rotated = cv2.warpAffine(img, M, (iw, ih))          
        
        return rotated
    # If cannot rotate the image, return input image directly
    return img



# Pre-processing of images for FaceNet model
def preProcess(img,requiredSize = (224, 224)):
    img = Image.fromarray(img)
    img = img.resize(requiredSize)
    img = np.asfarray(img)
    return img

# Face detection with MediaPipe (input: images, output: just faces)
def extractFaces(img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    faces = results.detections # Number of faces
    
    if faces == None:
        return None
    
    elif len(faces) == 0:
        return None
    
    elif len(faces) > 1:
        return None

    elif len(faces) == 1:
        for id, detection in enumerate(faces):
            h, w, c  = img.shape
            bbox = detection.location_data.relative_bounding_box
            bbox = int(bbox.xmin * w), int(bbox.ymin * h), \
                   int(bbox.width * w), int(bbox.height * h)
    
            #face = img[bbox[1]-30:bbox[1]+bbox[3]+30, bbox[0]-30:bbox[0]+bbox[2]+30]
            #face = preProcess(face)
    
            return [(bbox[1]-50),(bbox[1]+bbox[3]+30),(bbox[0]-50), (bbox[0]+bbox[2]+50)] 





