import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.layers import Dropout,Flatten
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
optim = tf.keras.optimizers.Adam()
from tensorflow.python.keras.optimizers import *
#Parameters#
path="myData"#folder with all class folders
labelFile='labels.csv'#file with all names of classes
batch_size_val=56 #how many process together
steps_per_epoch_val=200
epochs_val=4
imageDimesions=(32,32,3)
testRatio=0.2 #if 1000 images split will 200 for testing
validationRatio=0.2 #if 1000 images 20% of remaining 800 will be 160 for validation

#importing of the images
count = 0
images =[]
classNo=[]
myList=os.listdir(path)
print("Total Classes Detected:",len(myList))
noOfClasses=len(myList)
print("Importing Classes..")
for x in range(0,len(myList)):
    myPicList=os.listdir(path+"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y)
        images.append(curImg)
        classNo.append(count)
    print(count,end=" ")
    count+=1

print(" ")
images = np.array(images)
classNo= np.array(classNo)

#split data#

X_train , X_test,y_train, y_test = train_test_split(images,classNo, test_size = testRatio)
X_train , X_validation,y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

#x_train = Array for ımages to train
#x_Test= Corresponding Class ıd

### TO CHECK IF NUMBER OF IMAGES MATCHES TO NUMBER OF LABELS FOR EACH DATA SET
print("DATA SHAPES")
print("Train",end="");print(X_train.shape,y_train.shape)
print("Validation",end="");print(X_validation,y_validation.shape)
print("Test",end="");print(X_test,y_test.shape)
assert(X_train.shape[0] == y_train.shape[0]),"The number of images in not equal to the number of lables in training set"
assert(X_validation.shape[0] == y_validation.shape[0]),"The number of images in not equal to the number of lables in validation set"
assert(X_test.shape[0] == y_test.shape[0]),"The number of images in not equal to the number of lables in the test set"
assert(X_train.shape[1:] == (imageDimesions)),"The dimensions of the Training images are wrong"
assert(X_validation.shape[1:] == (imageDimesions)),"The dimesions of the Validation images are wrong"
assert(X_test.shape[1:] == (imageDimesions)),"The dimensions of the Test images are wrong"

###READ CSV FILE
data=pd.read_csv(labelFile)
print("data shape",data.shape,type(data))
## DISPLAY SOME SAMPLES IMAGES OF ALL THE CLASSES
num_of_samples=[]
cols=5
num_classes= noOfClasses
fig,axs = plt.subplots(nrows=num_classes,ncols=cols, figsize=(5,300))
fig.tight_layout()
for i in range(cols):
    for j,row in data.iterrows():
        x_selected = X_train[y_train==j]
        axs[j][i].imshow(x_selected[random.randint(0,len(x_selected)-1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i ==2:
            axs[j][i].axis("off")
            num_of_samples.append(len(x_selected))



### DISPLAY A BAR CHART SHOWING NO OF SAMPLES FOR EACH CATEGORY
print(num_of_samples)
plt.figure(figsize = (12,4))
plt.bar(range(0, num_classes),num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlable("Class number")
plt.ylable("Number of images")
plt.show()

### PREPROCESSING THE IMAGES
def grayscale(img):
     img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     return img
def equalize(img):
    img=cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img =grayscale(img)
    img= equalize(img)
    img=img/255
    return img

X_train=np.array(list(map(preprocessing,X_train)))
X_validation=np.array(list(map(preprocessing,X_validation)))
X_test=np.array(list(map(preprocessing,X_test)))
cv2.imshow("GrayScale Images",X_train[random.randint(0,len(X_train)-1)])#TO CHECK IF THE TRAINING IS DONE PROPERLY



X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_validation=X_validation.reshape(X_validation[0],X_validation.shape[1],X_validation.shape[2],1)
X_test=X_test.reshape(X_test.reshape(0),X_test.shape[1],X_test[2],1)



dataGen= ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10)

dataGen.fit(X_train)
batches=dataGen.flow(X_train,y_train,batch_size=20)
X_batch,y_batch= next(batches)

fix,axs=plt.subplots(1,15,figsize = (20,5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimesions[0],imageDimesions[1]))
    axs[i].axis('off')

plt.show()

y_train= to_categorical(y_train,noOfClasses)
y_validation=to_categorical(y_validation,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)


def myModel():
    no_Of_Filters=60
    size_of_Filter=(5,5)

    size_of_Filter_2=(3,3)
    size_of_pool=(2,2)
    no_Of_Nodes= 500
    model=Sequential()
    model.add((Conv2D(no_Of_Filters,size_of_Filter,input_shape=(imageDimesions[0],imageDimesions[1],1),activation='relu')))
    model.add((Conv2D(no_Of_Filters,size_of_Filter,activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add((Conv2D(no_Of_Filters//2, size_of_Filter_2,activation='relu')))
    model.add((Conv2D(no_Of_Filters//2,size_of_Filter_2,activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout)

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation='softmax'))


    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model


model=myModel()
print(model.summary())
history=model.fit_generator(dataGen.flow(X_train,y_train,batch_size=batch_size_val),steps_per_epoch_val=steps_per_epoch_val,epochs=epochs_val,validation_data=(x_selected))

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlable('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlable('epoch')
plt.show()

score= model.evaluate(X_test,y_test,verbose=0)
print('Test Score:',score[0])
print('Test Accuracy:',score[1])

pickle_out=open("model_trained.p","wb")
pickle.dump(model.pickle_out)
pickle_out.close()
cv2.waitKey(0)