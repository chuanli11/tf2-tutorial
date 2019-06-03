import datetime
import csv
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import applications
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


import resnet
import cv2
HEIGHT = 224
WIDTH = 224
NUM_CHANNELS = 3
NUM_CLASSES = 120


NUM_GPUS = 2
BS_PER_GPU = 8
NUM_EPOCHS = 20

BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 10), (0.01, 15)]
L2_WEIGHT_DECAY = 2e-4

MEAN = [103.939, 116.779, 123.68]

def preprocess(x, y):
  x = tf.compat.v1.read_file(x)
  x = tf.image.decode_jpeg(x, dct_method="INTEGER_ACCURATE")
  x = tf.compat.v1.image.resize_images(x, (HEIGHT, WIDTH))
  x = tf.cast(x, tf.float32)
  x = x - MEAN
  return x, y


def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(
        x, HEIGHT + 32, WIDTH + 32)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
    x = tf.image.random_flip_left_right(x)
    return x, y	


# def schedule(epoch):
#   initial_learning_rate = BASE_LEARNING_RATE * BS_PER_GPU / 128
#   learning_rate = initial_learning_rate
#   for mult, start_epoch in LR_SCHEDULE:
#     if epoch >= start_epoch:
#       learning_rate = initial_learning_rate * mult
#     else:
#       break
#   tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
#   return learning_rate

train_file = "/home/ubuntu/demo/data/StanfordDogs120/train.csv"
test_file = "/home/ubuntu/demo/data/StanfordDogs120/eval.csv"

def load_csv(file):
  dirname = os.path.dirname(file)
  images_path = []
  labels = []
  with open(file) as f:
    parsed = csv.reader(f, delimiter=",", quotechar="'")
    for row in parsed:
      images_path.append(os.path.join(dirname, row[0]))
      labels.append(int(row[1]))
  return images_path, labels


train_images_path, train_labels = load_csv(train_file)
test_images_path, test_labels = load_csv(test_file)

NUM_TRAIN_SAMPLES = len(train_images_path)
NUM_TEST_SAMPLES = len(test_images_path)

train_loader = tf.data.Dataset.from_tensor_slices((train_images_path, train_labels))
test_loader = tf.data.Dataset.from_tensor_slices((test_images_path, test_labels))

train_loader = train_loader.map(preprocess).map(augmentation).shuffle(NUM_TRAIN_SAMPLES).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)
test_loader = test_loader.map(preprocess).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)

input_shape = (HEIGHT, WIDTH, 3)
img_input = tf.keras.layers.Input(shape=input_shape)
opt = keras.optimizers.RMSprop()


if NUM_GPUS == 1:
    model = keras.models.load_model('ResNet50.h5')
    model.trainable = False
    x = model.layers[-3].output
    my_output = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    my_output = tf.keras.layers.Dropout(0.5)(my_output)
    my_output = tf.keras.layers.Dense(640,
                                kernel_initializer='he_normal',
                                kernel_regularizer=
                                tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                bias_regularizer=
                                tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                name='fc10')(my_output)
    my_output = tf.keras.layers.BatchNormalization()(my_output)
    my_output = tf.keras.layers.Activation("relu", name='myactivation')(my_output)
    my_output = tf.keras.layers.Dropout(0.5)(my_output)
    my_output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax',
                                kernel_initializer='he_normal',
                                kernel_regularizer=
                                tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                bias_regularizer=
                                tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                name='prediction')(my_output)      
    mymodel = tf.keras.models.Model(model.input, my_output, name='my')
    mymodel.compile(
              optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  
else:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
      model = keras.models.load_model('ResNet50.h5')
      model.trainable = False
      x = model.layers[-3].output
      my_output = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
      my_output = tf.keras.layers.Dropout(0.5)(my_output)
      my_output = tf.keras.layers.Dense(640,
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=
                                  tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                  bias_regularizer=
                                  tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                  name='fc10')(my_output)
      my_output = tf.keras.layers.BatchNormalization()(my_output)
      my_output = tf.keras.layers.Activation("relu", name='myactivation')(my_output)
      my_output = tf.keras.layers.Dropout(0.5)(my_output)
      my_output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=
                                  tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                  bias_regularizer=
                                  tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                  name='prediction')(my_output)      
      mymodel = tf.keras.models.Model(model.input, my_output, name='my')
      mymodel.compile(
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

# lr_schedule_callback = keras.callbacks.LearningRateScheduler(schedule)

mymodel.fit(train_loader,
          epochs=NUM_EPOCHS,
          validation_data=test_loader,
          validation_freq=1,
          callbacks=[tensorboard_callback])
mymodel.evaluate(test_loader)
