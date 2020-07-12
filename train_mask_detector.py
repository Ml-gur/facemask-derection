#train a face-mask detector using OpenCV,Keras,Tensorflow and deep learning

#importing packages
###all these packages allow data augmentation,loading the mobilenetv2,building new fully-connected head,preprocessing,loading image data

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.preprocessing.image import load_image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

#creaing an argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
ap.add_argument("-p","--plot",type=str,default="plot.png",
                help="path to output loss/accuracy plot")

ap.add_argument("-m","--model",type=str,default = "mask_detection.model",
                help="path to output face mask detector model")

args = vars(ap.parse_args())

#initializing learning rate,number of epochs to train for and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

#grabing the list of images in dataset directory,then initializing the list of data
print("INFO loading images...")
imagePaths = list(paths.list_images(args["dataset"]))#grabing all image paths
data = []
labels = []

#looping over the image paths

for imagePath in imagePaths:
      #extracting the class label from filename
      label = imagePath.split(os.path.sep)[-2]

      #load the input image(224x224)
      image = load_img(imagePath,target_size = (224,224))#resizing my images
      image = img_to_array(image)
      image = preprocess_input(image)

      #updating the data labels list
      data.append(image)
      labels.append(label)

      #converting the data and labels to Numpy arrays
data = np.array(data,dtype="float32")
labels = np.array(labels)

#coding on labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#here i will divide the data into training and testing data,80% for training and the rest for testing
(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size = 0.20,
                                        stratify=labels,random_state=42)

#creating training image generator
aug= ImageDataGenerator(
    rotation_range=20,
    zoom_range = 0.15,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.15,
    horizontal_flip = True,
    fill_mode="nearest")

#loading the mobilenetv2 network
baseModel = MobileNetV2(weights="imagenet",include_top=False,
                        input_tensor = Input(shape=(224,224,3)))

#constructing the head of the model that will be on top of base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128,activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2,activation="softmax")(headModel)

#the model i will train after placing head ontop of base model
model = Model(inputs=baseModel.input,outputs=headModel)

#looping over al the layers and freeze them all so they won get updated during first training
for layer in baseModel.layers:
    layer.trainable = False

#compiling the model i created above
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR,decay=INIT_LR / EPOCHS)#Adam is an optimizer
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

#training the head of my network
print("INFO] training head...")
H = model.fit(
    aug.flow(trainX,trainY,batch_size = BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data = (testX,testY),
    validation_steps = len(testX) // BS,
    epochs = EPOCHS)


#after training
#i now make predictions on the testing data set
print("[INFO] evaluating network...")
#predIdxs = model.predict(testX,batch_size=BS)

#finding index of each image in testing set
#predIdxs = np.argmax(predIdxs,axis=1)

print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

#he classifiation report
#print(classification_report(testY.argmax(axis=1),predIxs,
 #                           target_names = lb.classes_))

#serializing he model to disk
print("[INFO] saving mask detector model...")
model.save(args["model"],save_format="h5")

#ploting the trainingloss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,N),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,N),H.history["accuracy"],label="train_acc")
plt.plot(n.arange(0,N),H.history["val_accuracy"],label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])





































































                        







































































      










































