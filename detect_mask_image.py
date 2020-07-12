#detecting mask using opencv

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

#constructing argument parser to parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,
                 help="path to input image")#--image contains my input image for inference

ap.add_argument("-f","--face",type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m","--model",type=str,
                default="mask_detection.model",
                help="path to trained face mask detector model")
ap.add_argument("-c","--confidence",type=float,default=0.5,
                help="minimu probability to filter weak detections")
args = vars(ap.parse_args())
                
#loading my face detector
#load the sereialized face detector model from disk
print("[INFO] loading face detctor model...")
prototxtPath = os.path.sep.join([args["face"],"deploy.prottxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_inter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath,weightsPath)
#loading face mask detector model

print("[INFO] loading face mask detector model...")
model = load_image(args["model"])

#preprosessing an input image
#loading the input image from disk, cloning it and grabbing the image spatial diemnsions
image = cv2.imread(args["image"])
orig = image.copy()
(h,w) = image.shape[:2]

#constructing a blob for the image
blob = cv2.dnn.blobFromImage(image,1.0,(300,300),(104.0,177.0,123.0))

#passing the blob through the network and obtain the face detections
print("[INFO] computing face detection...")
net.setInput(blob)
detections = net.forwad()

#ensuring the predicted faces meet the confidence threshhold
#looping over the detections
for i in range(0,detections.shape[2]):
    #extracting the confidence 'probability'
    confidence = detections[0,0,i,2]

    #filtering out weak detections by ensuring the confidence is great
    if confidence > args["confidence"]:
        #compute the x,y coordiantes
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (startX,startY,endX,endY) = box.astype("int")
        #ensuring the bunding box fall within dimenso=ions of frame
        (startX,startY) = (max(0,startX),max(0,startY))
        (endX,endY) = (min(w - 1,endX),min(h -1,endY))

        #extracting the face ROI(region of interest) convert it to RGB and resize it
        face = image[startY:endY,startX:endX]
        face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        face = cv2.resize(face,(224,224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face,axis=0)

        #passing the face through model to determine if face has mask
        (mask,withoutMask) = model.predict(face)[10]

        #displaying the result
        #determining the class label and color i'l use to draw
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0,255,0) if label == "Mask" else (0,0,255)

        #including the probability in label
        label = "{}: {:.2f}%".format(label,max(mask,withoutMask) * 100)

        #display the label and bounding box rectangle on output fram
        cv2.putText(image, label, (startX,startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
        cv2.rectangle(image,(startX,startY),(endX,endY),color,2)


#show output
cv2.imshow("output",image)
cv2.waitKey(0)

    




















        































