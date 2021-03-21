import cv2

path="Resources/lena.png"
img=cv2.imread(path)
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(7,7),0)
imgCanny=cv2.Canny(imgBlur,100,200)

cv2.imshow("lena",img)
cv2.imshow("Grayscale",imgGray)
cv2.imshow("imgBlur",imgBlur)
cv2.imshow("img canny",imgCanny)
cv2.waitKey(0)