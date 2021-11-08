#Julian Blanco
from keras.layers import * 
from keras.models import Sequential
from keras.optimizer_v2 import adam
from keras.optimizers import adam_v2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as ply
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import os

labels = ["DVI", "HDMI"]
img_size = 256

def get_data(data_dir):
    data = [] #Data Array
    for label in labels: # For the amount of items in the labels
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

train = get_data("PortellerPhotos/Training")
val = get_data("PortellerPhotos/Testing")

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)


x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

print(train.shape)

plt.figure(figsize = (5,5))
plt.imshow(train[5][0])
plt.title(labels[train[0][1]])
plt.show()

model = Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(img_size,img_size, 3)))
model.add(Dense(32, activation='relu'))
model.add(Dense(128))
model.add(Dense(1))
model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy")

model.fit(x_train, y_train, epochs=2)