import cv2
import numpy as np

frameWidth= 640
frameHeight = 480

cap= cv2.VideoCapture(0)

cap.set(3, frameWidth)
cap.set(4, frameHeight)


while True:
    _, img= cap.read()
    imgHsv=cv2.cvtColor()
    cv2.imshow()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
