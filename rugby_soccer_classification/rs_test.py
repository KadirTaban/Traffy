import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf

import cv2
import os
import numpy as np

##loading the data
labels=['rugby','soccer']

img_size = 224

def get_data(data_dir) -> object:#Directory path to use when loading key files
    """

    :rtype: object
    """
    data=[]
    for label in labels:
        path = os.path.join(data_dir,label)
        class_num=labels.index(label)

        for img in os.listdir(path):
            try:
                img_arr=cv2.imread(os.path.join(path,img))[...,::-1]#convert BGR to RGB format
                resized_arr=cv2.resize(img_arr,(img_size,img_size))#Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e :
                print(e)

    return np.ndarray(object,data)

x_train= get_data('/home/mountainlabs/OpenCV-basics/rugby_soccer_classification/input/train')
validation= get_data('/home/mountainlabs/OpenCV-basics/rugby_soccer_classification/input/test')

l=[]

for i in x_train:
    if(i[1] == 0):
        l.append("rugby")
    else:
        l.append("soccer")

sns.set_style('darkgrid')
sns.countplot(l)
