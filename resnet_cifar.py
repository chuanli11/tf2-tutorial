import datetime
import math 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import TensorBoard
import resnet

from tensorflow.python.keras.optimizer_v2 import (gradient_descent as
                                                  gradient_descent_v2)

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
num_classes = 10

num_gpus = 2
num_train_samples = 50000
bs_per_gpu = 128
num_epochs = 10

BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 4), (0.01, 8), (0.001, 10)]

def preprocess(x, y):
  x = tf.image.per_image_standardization(x)
  return x, y


def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(
        x, HEIGHT + 8, WIDTH + 8)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
    x = tf.image.random_flip_left_right(x)
    return x, y	


def schedule(epoch):
  initial_learning_rate = BASE_LEARNING_RATE * bs_per_gpu / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  return learning_rate


(x,y), (x_test, y_test) = keras.datasets.cifar10.load_data()

train_loader = tf.data.Dataset.from_tensor_slices((x,y))
test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

tf.random.set_seed(22)
train_loader = train_loader.map(augmentation).map(preprocess).shuffle(num_train_samples).batch(bs_per_gpu * num_gpus, drop_remainder=True)
test_loader = test_loader.map(preprocess).batch(bs_per_gpu * num_gpus, drop_remainder=True)

opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

if num_gpus == 1:
    model = resnet.resnet56(classes=num_classes)
    model.compile(
              optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
else:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
	    model = resnet.resnet56(classes=num_classes)
	    model.compile(
                optimizer=opt,
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])  

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = TensorBoard(
  log_dir=log_dir,
  update_freq='batch',
  histogram_freq=1)

lr_schedule_callback = keras.callbacks.LearningRateScheduler(schedule)

model.fit(train_loader,
          epochs=num_epochs,
          validation_data=test_loader,
          validation_freq=1,
          callbacks=[tensorboard_callback, lr_schedule_callback])
model.evaluate(test_loader)

model.save('model.h5')

new_model = keras.models.load_model('model.h5')
 
new_model.evaluate(test_loader)