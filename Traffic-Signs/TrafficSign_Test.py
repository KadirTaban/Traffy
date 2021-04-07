import numpy as np
import cv2
import pickle


frameWidth = 640
frameHeight = 480  #camera resolution
brightness = 180
threshOld = 0.75 #probability threshold
font = cv2.FONT_HERSHEY_SIMPLEX
#Setup video Camera
cap =cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,brightness)
#Import the trained model

pickle_in=("model_trained.p","rb") #read byte
model=pickle.load(pickle_in)

def grayscale(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img=cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img=grayscale(img)
    img=equalize(img)
    img=img/255
    return img

def getClassname=