import datetime
import math 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import TensorBoard


HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
INIT_LR = 1e-3 
num_train_samples = 40000
bs_per_gpu = 128
num_gpus = 1
num_epochs = 10

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, update_freq, histogram_freq):  # add other arguments to __init__ if you need
        super(LRTensorBoard, self).__init__(log_dir=log_dir,
        				 update_freq=update_freq,
        				 histogram_freq=histogram_freq)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': self.model.optimizer.lr})
        super(LRTensorBoard, self).on_epoch_end(epoch, logs)
        

def preprocess(x, y):
  image = tf.image.per_image_standardization(x)
  return x, y


def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(
        x, HEIGHT + 8, WIDTH + 8)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])

    # Randomly flip the image horizontally.
    x = tf.image.random_flip_left_right(x)

    return x, y	


def schedule(epoch):
	initial_lrate = INIT_LR
	drop = 0.5
	epochs_drop = 2.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


def VGG16(input_shape):
  # Do not use subclass for easier save/load model and print summary
  weight_decay = 0.000
  num_classes = 10

  model = models.Sequential()

  flag_BN = True

  model.add(layers.Conv2D(64, (3, 3), padding='same',
                   input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  if flag_BN:
    model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  if flag_BN:
    model.add(layers.BatchNormalization())
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))

  model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  if flag_BN:
    model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  if flag_BN:
    model.add(layers.BatchNormalization())

  model.add(layers.MaxPooling2D(pool_size=(2, 2)))

  model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  if flag_BN:
    model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  if flag_BN:
    model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  if flag_BN:
    model.add(layers.BatchNormalization())
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))


  model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  if flag_BN:
    model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  if flag_BN:
    model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  if flag_BN:
    model.add(layers.BatchNormalization())

  model.add(layers.MaxPooling2D(pool_size=(2, 2)))


  model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  if flag_BN:
    model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  if flag_BN:
    model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  if flag_BN:
    model.add(layers.BatchNormalization())

  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Dropout(0.5))

  model.add(layers.Flatten())
  model.add(layers.Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  if flag_BN:
    model.add(layers.BatchNormalization())

  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(num_classes))
  model.add(layers.Activation('softmax'))  

  return model


(x,y), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_val = x[num_train_samples:, :]
y_val = y[num_train_samples:, :]

x = x[:num_train_samples, :]
y = y[:num_train_samples, :]

train_loader = tf.data.Dataset.from_tensor_slices((x,y))
val_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

tf.random.set_seed(22)
train_loader = train_loader.map(augmentation).map(preprocess).shuffle(num_train_samples).batch(bs_per_gpu * num_gpus)
val_loader = val_loader.map(preprocess).batch(bs_per_gpu * num_gpus)
test_loader = test_loader.map(preprocess).batch(bs_per_gpu * num_gpus)


if num_gpus == 1:
    model = VGG16([32, 32, 3])
    model.compile(
              optimizer=keras.optimizers.Adam(learning_rate=INIT_LR),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = LRTensorBoard(
  log_dir=log_dir,
  update_freq='batch',
  histogram_freq=1)

lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)


model.fit(train_loader,
          epochs=num_epochs,
          validation_data=val_loader,
          validation_freq=1,
          callbacks=[tensorboard_callback, lr_schedule_callback])
model.evaluate(test_loader)