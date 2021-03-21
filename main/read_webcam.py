import cv2

frameWidth=540
frameHeight=400


cap= cv2.VideoCapture(0)

cap.set(3,frameWidth)
cap.set(4,frameHeight)

while True:
    sucess,img=cap.read()
    img=cv2.resize(img,(frameWidth,frameHeight))
    cv2.imshow("Video",img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break