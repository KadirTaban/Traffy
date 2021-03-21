import cv2

path="Resources/road.jpg"
img=cv2.imread(path)
print(img.shape)

width , height = 1000,1000
imgResize=cv2.resize(img,(width,height))
print(imgResize.shape)
imgCropped=img[300:500,430:1000]
imgCropResize = cv2.resize(imgCropped,(img.shape[0],img.shape[0]))
#cv2.imshow("road",img)
#cv2.imshow("road",imgResize)
cv2.imshow("roadcropped",imgCropped)
cv2.imshow("road cropped resized",imgCropResize)
cv2.waitKey(0)