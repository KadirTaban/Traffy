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

def getClassname(classNo):
    if classNo==0:return"Speed limit(80km/h)"
    elif classNo==1:return "No Passing"
    elif classNo == 2:return "Stop"
    elif classNo== 3:return "Yield"


while True:
    #Read Image
    success, imgOriginal=cap.read()
    #Process Image
    img=np.asarray(imgOriginal)
    img=cv2.resize(img,(32,32))
    img=preprocessing(img)
    cv2.imshow("Processed Image",img)
    img=img.reshape(1,32,32,1)

    cv2.putText(imgOriginal,"CLASS",(20,35),font,0.75,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(imgOriginal,"Probability",(20,75),font,0.75,(0,0,255),2,cv2.LINE_AA)
    #PREDICT IMAGE
    predictions=model.predict(img)
    classIndex=model.predict_classes(img)
    probabilityValue=np.amax(predictions)
    if probabilityValue > threshOld:
        #print(getClassName(classIndex))
        cv2.putText(imgOriginal,str(classIndex)+""+str(getClassname(classIndex)),(120,35),font,0.75,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(imgOriginal,str(round(probabilityValue*100,2))+"%",(100,75),font,0.75,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow("Result",imgOriginal)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break