import cv2

path="Resources/lena.png"
img=cv2.imread(path)
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("lena",img)
cv2.imshow("Grayscale",imgGray)
cv2.waitKey(0)