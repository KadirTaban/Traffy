import cv2
import numpy as np

frameWidth=640
frameHeight = 480

cap=cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
def empty(a):
    pass

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV",640,480)
cv2.createTrackbar("Hue Min","HSV",0,179,empty)

while True:
    _, img=cap.read()
    imgHsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    cv2.imshow('Original',img)
    cv2.imshow('HSC Color Space',imgHsv)

    if cv2.waitKey(1)& 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()