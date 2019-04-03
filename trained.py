from keras.models import Model
from keras.layers import Input, Dense

from keras.models import model_from_json
import cv2
import numpy as np
import os
from random import shuffle
import tensorflow as tf

from keras.layers import *
from keras.optimizers import Adam
import sys

from keras.models import load_model
import cv2
import numpy as np
import pyttsx3

data = sys.argv[1]

model = load_model("model.h5")

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])



img = cv2.imread(data,cv2.IMREAD_COLOR)
#img = cv2.imread("/home/nick/slam/vid7/clear.0.jpg",cv2.IMREAD_COLOR)
img = cv2.resize(img,(32,32))
img = np.reshape(img,(1,32,32,3))

classes = model.predict(img)

if np.argmax(classes) == 0:
  pred = "obstruction_left"
elif np.argmax(classes) == 1:
  pred = "clear"
elif np.argmax(classes) == 2:
  pred = "obstruction_ahead"
elif np.argmax(classes) == 3:
  pred = "obstruction_right"


def onStart(name):
  print('starting',name)
def onWord(name,location,length):
  print('word',name,location,length)
def onEnd(name, completed):
  print('finishing',name,completed)

engine=pyttsx3.init()
engine.connect('started-utterance',onStart)
engine.connect('started-word',onWord)
engine.connect('finished-utterance',onEnd)
engine.say(pred)
engine.runAndWait()
