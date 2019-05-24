import sys
import  numpy as np
import datetime

import  tensorflow as tf
from tensorflow import  keras
from tensorflow.keras import models, layers, regularizers
from tensorflow.python.keras.optimizer_v2 import (gradient_descent as
                                                  gradient_descent_v2)
import resnet
import keras_common

num_classes = 10
num_train_samples = 40000
bs_per_gpu = 128
num_epochs = 10
num_gpus = 1

def normalize(X_train, X_test):
    X_train = X_train / 255.
    X_test = X_test / 255.

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    print('mean:', mean, 'std:', std)
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test

def prepare_cifar(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    return x, y

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


def learning_rate_schedule(current_epoch,
                           current_batch,
                           batches_per_epoch,
                           batch_size):
  """Handles linear scaling rule and LR decay.

  Scale learning rate at epoch boundaries provided in LR_SCHEDULE by the
  provided scaling factor.

  Args:
    current_epoch: integer, current epoch indexed from 0.
    current_batch: integer, current batch in the current epoch, indexed from 0.
    batches_per_epoch: integer, number of steps in an epoch.
    batch_size: integer, total batch sized.

  Returns:
    Adjusted learning rate.
  """
  del current_batch, batches_per_epoch  # not used
  initial_learning_rate = keras_common.BASE_LEARNING_RATE * batch_size / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if current_epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  return learning_rate


print('loading data...')
(x,y), (x_test, y_test) = keras.datasets.cifar10.load_data()
x, x_test = normalize(x, x_test)

x_val = x[num_train_samples:, :]
y_val = y[num_train_samples:, :]

x = x[:num_train_samples, :]
y = y[:num_train_samples, :]

train_loader = tf.data.Dataset.from_tensor_slices((x,y))
val_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

tf.random.set_seed(22)
train_loader = train_loader.map(prepare_cifar).shuffle(num_train_samples).batch(bs_per_gpu * num_gpus)
val_loader = val_loader.map(prepare_cifar).batch(bs_per_gpu * num_gpus)
test_loader = test_loader.map(prepare_cifar).batch(bs_per_gpu * num_gpus)      

optimizer = keras_common.get_optimizer()

# callbacks = keras_common.get_callbacks(
#     learning_rate_schedule, num_train_samples)

if False:
  for xx, yy in train_loader:
    print(xx.shape)
    print(yy.shape)
    break

if num_gpus == 1:
    model = resnet.resnet56(classes=num_classes)
    model.compile(
              optimizer=optimizer,
              # optimizer=gradient_descent_v2.SGD(learning_rate=1e-3, momentum=0.9),
              # optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  
    # model = VGG16([32, 32, 3])
    # model.compile(
    #           optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    #           loss='sparse_categorical_crossentropy',
    #           metrics=['accuracy'])   

else:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = resnet.resnet56(classes=num_classes)
        model.compile(
                  optimizer=gradient_descent_v2.SGD(learning_rate=0.1, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])   

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
  log_dir=log_dir,
  update_freq=bs_per_gpu * num_gpus * 10,
  histogram_freq=1)

model.fit(train_loader,
          epochs=num_epochs,
          validation_data=val_loader,
          validation_freq=1,
          callbacks=[tensorboard_callback])
model.evaluate(test_loader)

# Save & load weights
# Cannot save model configuration: http://ashokrahulgade.com/coding/keras/Module1.html
# Save weights to disk
model.save('model.h5')

new_model = keras.models.load_model('model.h5')
# Result will be slightly different if training uses multiple-gpus
# Related to batch normalization layer   
new_model.evaluate(test_loader)