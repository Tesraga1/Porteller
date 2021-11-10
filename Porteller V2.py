#Julian Blanco
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import *
import tensorflow
import os
from keras.preprocessing.image import *
import matplotlib.pyplot as plt
import numpy as np

dataset_directory = "PortellerPhotos/"
Test_path = "TestingPictures/Display_test.jpg"
img_dimension = 255
epochs = 30
num_classes = 4
batch_data = 16
epochs_range = range(epochs)

train_data = image_dataset_from_directory(
    dataset_directory,
    validation_split=0.4,
    subset="training",
    seed=123,
    image_size=(img_dimension, img_dimension),
    batch_size=batch_data,
)

val_ds = image_dataset_from_directory(
  dataset_directory,
  validation_split=0.4,
  subset="validation",
  seed=123,
  image_size=(img_dimension, img_dimension),
  batch_size=batch_data,
)

classnames = train_data.class_names
print(classnames)

for image_batch, labels_batch in train_data:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential(
  [
    RandomFlip("horizontal",
                      input_shape=(img_dimension,
                                  img_dimension,
                                  3)),
    RandomRotation(0.1),
    RandomZoom(0.1),
  ]
)

plt.figure()
for images, labels in train_data.take(1):
  for i in range(9):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(classnames[labels[i]])
    plt.axis("off")



model = Sequential([
  data_augmentation,
  Rescaling(1./255),
  Conv2D(16, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Conv2D(32, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Conv2D(64, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Dropout(.5),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


history = model.fit(train_data, validation_data=val_ds, epochs=epochs)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']



plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


img = keras.utils.load_img(
    Test_path, target_size=(img_dimension, img_dimension)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(predictions[0])
print("Predicted Label: " + str(np.argmax(predictions[0])))

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(classnames[np.argmax(score)], 100 * np.max(score))
)

model.save("PortellerModel")