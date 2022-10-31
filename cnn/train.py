#coding: UTF-8
import numpy as np
import tensorflow as tf
import numpy
from tensorflow.keras import datasets, layers, models
import preprocessing

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28,1))
test_images = test_images.reshape((10000, 28, 28,1))

data_path = "./image_data"
new_train_images, new_train_labels = preprocessing.read_image(data_path)

train_labels_categorical = tf.keras.utils.to_categorical(new_train_labels)
test_labels_categorical = tf.keras.utils.to_categorical(test_labels)


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #rotation_range=10,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    #zoom_range = [1,1.2],
    rescale=1./255)

batch_size = 32
data_num = len(new_train_images)


checkpoint_path = './model/handwriting_cnn.h5'

model.summary()

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=False,
                                                 verbose=1)

datagen.fit(new_train_images)

model.fit(datagen.flow(new_train_images, train_labels_categorical, batch_size = batch_size, shuffle=True), 
          steps_per_epoch = data_num // batch_size,
          epochs=20, 
          callbacks=[cp_callback],
          verbose=1)



