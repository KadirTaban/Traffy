import cv2
import numpy as np

img= cv2.imread("Resources/iskambil.jpg")

width, height= 450,550

pts1 = np.float32([[134,194],[243,224],[129,339],[245,355]])
pts2= np.float32([[0,0],[450,0],[0,550],[450,550]])
matrix= cv2.getPerspectiveTransform(pts1,pts2)#ex:https://pysource.com/2018/02/14/perspective-transformation-opencv-3-4-with-python-3-tutorial-13/
imgOutput=cv2.warpPerspective(img,matrix,(width,height))


for x in range(0,4):


    cv2.circle(img,(pts1[x][0],pts1[x][1]),15,(0,255,0),cv2.FILLED)


cv2.imshow("Original Image",img)
cv2.imshow("Output Image",imgOutput)
cv2.waitKey(0)