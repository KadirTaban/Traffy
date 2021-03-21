import cv2

path="Resources/road.jpg"
img=cv2.imread(path)


cv2.imshow("road",img)

cv2.waitKey(0)