import cv2
import numpy as np

kernel = np.ones((5,5),np.uint8)
print(kernel)
path="Resources/lena.png"
img=cv2.imread(path)
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(7,7),0)
imgCanny=cv2.Canny(imgBlur,100,200)
imgDilation=cv2.dilate(imgCanny,kernel, iterations = 1)
#dilasyon işleminde canny formatındaki image'de çizgileri kalınlaştırır.
imgErosed = cv2.erode(imgDilation,kernel,iterations= 1)
#erozyon işleminde canny formatındaki image'de çizgileri aşındırır.

#cv2.imshow("lena",img)
#cv2.imshow("Grayscale",imgGray)
#cv2.imshow("imgBlur",imgBlur)
cv2.imshow("img canny",imgCanny)
cv2.imshow("img dilation",imgDilation)
cv2.imshow("img erosed",imgEroded)

cv2.waitKey(0)