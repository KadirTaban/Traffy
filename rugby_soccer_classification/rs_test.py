from typing import Union

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.python.keras.losses
from matplotlib.pyplot import plot
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
from pathlib import Path
import cv2
import os
import numpy as np

##loading the data
labels=['rugby','soccer']

img_size = 224

labels = ['rugby', 'soccer']
img_size = 224
def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    #return np.asarray(data,dtype=np.float64)
    return np.array(data,list)


## we can easily fetch our train and validation data
train = get_data("/home/mountainlabs/OpenCV-basics/rugby_soccer_classification/input/train")
validation= get_data("/home/mountainlabs/OpenCV-basics/rugby_soccer_classification/input/test")

#visualize the data
l=[]

for i in train:
    if(i[1] == 0):
        l.append("rugby")
    else:
        l.append("soccer")

#sns.set_style('darkgrid')
#sns.countplot(l)
##plt.plot(l)
##plt.style.use(['seaborn'])
#plt.figure(figsize = (5,5))
#plt.imshow(train[1][0])
#plt.title(labels[train[0][1]])
#plt.figure(figsize = (5,5))
#plt.imshow(train[-1][0])
##plt.title(labels[train[-1][1]])

##plt.show()

x_train= []
y_train= []
x_val= []
y_val= []

for feature, label in train :
    x_train.append(feature)
    y_train.append(label)

for feature, label in validation:
    x_val.append(feature)
    y_val.append(label)

# Normalize the data

x_train = np.array(x_train)/255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size,img_size,1)
y_train= np.array(y_train)

x_val.reshape(-1 , img_size,img_size,1)
y_val=np.array(y_val)
#Data augmentation on the train data

datagen= ImageDataGenerator(featurewise_center=False,#set input mean to 0 over the dataset
                            samplewise_center=False, # set each sample mean to 0
                            featurewise_std_normalization=False, #divide each input by its std
                            zca_whitening=False, #apply ZCA whitening
                            rotation_range=30, #randomly zoom image
                            width_shift_range=0.1, #randomly shift images horizontally(fraction of total width)
                            height_shift_range=0.1, #randomly shift images vertically(fraction of total height)
                            horizontal_flip=True, #randomly flip images
                            vertical_flip= False) #randomly flip images


datagen.fit(x_train)
#Let's define a simple CNN model with 3 Convolutional layers followed by max-pooling layers.

model=Sequential()
model.add(Conv2D(32,3,padding="same",activation="relu", input_shape=(224,224,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32,3,padding="same",activation="relu"))
model.add(MaxPooling2D())
model.add(MaxPooling2D())
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2,activation="softmax"))
model.summary()
#Let's compile the model using Adam as our optimizer and SparseCategoricalCrossentropy as the loss function
opt= Adam(lr=0.000001)
model.compile(optimizer=opt,loss=tensorflow.python.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
#Let's train our model for 500 epochs since our learning rate is very small
history=model.fit(x_train,y_train,epochs=500,validation_data=(x_val,y_val))
#Evaluationg the result
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(500)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()