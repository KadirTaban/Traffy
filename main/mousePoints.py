import cv2
import numpy as np

def mousePoints(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)

img = cv2.imread("Resources/iskambil.jpg")
cv2.imshow("Original image", img)
cv2.setMouseCallback("Original image", mousePoints)

cv2.waitKey(0)