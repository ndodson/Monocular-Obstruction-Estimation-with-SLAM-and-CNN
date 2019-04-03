import numpy as np
import sys
import shutil
import os
import time

#arg is from display.py
#arg is the current image grabbed from pygame.save(image)
tmp = sys.argv[1]
#take this image file that is passed into this file
#and store it into cnnTest directory
#in future it will be passed directly to the cnn
#$shutil.copy2(tmp, "cnnTest/")




#sys.stdout = open('test.txt','a')
#print(time.strftime("%I:%M:%S\n"))

#TODO
#make sure we don't grab black picture
	#various ways to do this I'm thinking

#current implementation of if else in display.py
#doesn't work... have to drop image1.jpg in this file


import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.models import model_from_json 

train_data = '/home/nick/slam/vid7/'
test_data = '/home/nick/slam/vid7/'






def one_hot_label(img):
   
    global ohl
    label = img.split('.')[0]
    if label == 'obstruction_right':
      ohl = np.array([0,0,0,1])
    elif label == 'obstruction_ahead':
      ohl = np.array([0,0,1,0])
    elif label == 'clear':
      ohl = np.array([0,1,0,0,])
    elif label == 'obstruction_left':
      ohl = np.array([1,0,0,0])
    return ohl


def train_data_with_label():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data,i)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:

            img = cv2.resize(img, (32,32))
            print(one_hot_label(i))
            train_images.append([np.array(img), one_hot_label(i)])
        else:
            print("image not loaded")
    shuffle(train_images)
    print(train_images)
    return train_images

def test_data_with_label():

    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data,i)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:

            img = cv2.resize(img, (32,32))
            test_images.append([np.array(img), one_hot_label(i)])
        else:
            print("image not loaded")
    shuffle(test_images)
    print(test_images)
    return test_images


def get_categorical_accuracy_keras(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)))

training_images = train_data_with_label()
testing_images = test_data_with_label()
tr_img_data = np.array([i[0] for i in training_images]).reshape(14,32,32,3)
tr_lbl_data = np.array([i[1] for i in training_images])
tst_img_data = np.array([i[0] for i in testing_images]).reshape(14,32,32,3)
tst_lbl_data = np.array([i[1] for i in testing_images])

print(tr_img_data.shape)
print(tr_lbl_data.shape)


model = Sequential()

model.add(InputLayer(input_shape=(32,32,3)))
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=1,padding='same',activation='tanh'))
model.add(MaxPool2D(pool_size=(3,3),padding='same'))

model.add(Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',activation='tanh'))
model.add(MaxPool2D(pool_size=(3,3),padding='same'))


model.add(Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',activation='tanh'))
model.add(MaxPool2D(pool_size=(3,3),padding='same'))


model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512,activation='tanh'))
model.add(Dropout(rate=0.5))
model.add(Dense(4,activation='softmax'))
optimizer = Adam(lr=1e-3)

model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x=tr_img_data,y=tr_lbl_data,epochs=4)
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save("model.h5")


nick = cv2.imread("/home/nick/slam/vid7/clear.0.jpg",cv2.IMREAD_COLOR)
nick = cv2.resize(nick,(32,32))
nick = np.reshape(nick,(1,32,32,3))
print("Saved model to disk")

print(model.predict(nick))

