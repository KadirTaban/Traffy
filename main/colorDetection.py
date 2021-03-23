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
cv2.createTrackbar("Hue Max","HSV",179,179,empty)
cv2.createTrackbar("SAT Min","HSV",0,255,empty)
cv2.createTrackbar("SAT Max","HSV",255,255,empty)
cv2.createTrackbar("Value Min","HSV",0,255,empty)
cv2.createTrackbar("Value Max","HSV",255,255,empty)

while True:
    _, img=cap.read()
    imgHsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    cv2.namedWindow("HSV")
    cv2.resizeWindow("HSV", 640, 480)
    cv2.createTrackbar("Hue Min", "HSV", 0, 179, empty)
    cv2.createTrackbar("Hue Max", "HSV", 179, 179, empty)
    cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
    cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
    cv2.createTrackbar("Value Min", "HSV", 0, 255, empty)
    cv2.createTrackbar("Value Max", "HSV", 255, 255, empty)
    print(h_min)

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask=cv2.inRange(imgHsv,lower,upper)

    cv2.imshow('Original',img)
    cv2.imshow('HSC Color Space',imgHsv)

    if cv2.waitKey(1)& 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()