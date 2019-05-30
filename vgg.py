import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import models, layers, regularizers

import BN16

def VGG16(img_input, classes):
  # Do not use subclass for easier save/load model and print summary
  weight_decay = 0.000

  flag_BN = True
  flag_BN16 = False

  x = img_input
  x = layers.Conv2D(64, (3, 3), padding='same',
                    kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = layers.Activation('relu')(x)
  # if flag_BN:
  #   x = layers.BatchNormalization()(x)
  # if flag_BN16:
  #   x = BN16.BatchNormalizationF16()(x)
  x = layers.Dropout(0.3)(x)

  x = layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = layers.Activation('relu')(x)
  # if flag_BN:
  #   x = layers.BatchNormalization()(x)
  # if flag_BN16:
  #   x = BN16.BatchNormalizationF16()(x)    
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)

  x = layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = layers.Activation('relu')(x)
  # if flag_BN:
  #   x = layers.BatchNormalization()(x)
  # if flag_BN16:
  #   x = BN16.BatchNormalizationF16()(x)    
  x = layers.Dropout(0.4)(x)

  x = layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = layers.Activation('relu')(x)
  # if flag_BN:
  #   x = layers.BatchNormalization()(x)
  # if flag_BN16:
  #   x = BN16.BatchNormalizationF16()(x)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)

  x = layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = layers.Activation('relu')(x)
  if flag_BN:
    x = layers.BatchNormalization()(x)
  if flag_BN16:
    x = BN16.BatchNormalizationF16()(x)
  x = layers.Dropout(0.4)(x)

  x = layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = layers.Activation('relu')(x)
  if flag_BN:
    x = layers.BatchNormalization()(x)
  if flag_BN16:
    x = BN16.BatchNormalizationF16()(x)
  x = layers.Dropout(0.4)(x)

  x = layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = layers.Activation('relu')(x)
  if flag_BN:
    x = layers.BatchNormalization()(x)
  if flag_BN16:
    x = BN16.BatchNormalizationF16()(x)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)


  x = layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = layers.Activation('relu')(x)
  if flag_BN:
    x = layers.BatchNormalization()(x)
  if flag_BN16:
    x = BN16.BatchNormalizationF16()(x)
  x = layers.Dropout(0.4)(x)

  x = layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = layers.Activation('relu')(x)
  if flag_BN:
    x = layers.BatchNormalization()(x)
  if flag_BN16:
    x = BN16.BatchNormalizationF16()(x)

  x = layers.Dropout(0.4)(x)

  x = layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = layers.Activation('relu')(x)
  if flag_BN:
    x = layers.BatchNormalization()(x)
  if flag_BN16:
    x = BN16.BatchNormalizationF16()(x)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)


  x = layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = layers.Activation('relu')(x)
  if flag_BN:
    x = layers.BatchNormalization()(x)
  if flag_BN16:
    x = BN16.BatchNormalizationF16()(x)
  x = layers.Dropout(0.4)(x)

  x = layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = layers.Activation('relu')(x)
  if flag_BN:
    x = layers.BatchNormalization()(x)
  if flag_BN16:
    x = BN16.BatchNormalizationF16()(x)
  x = layers.Dropout(0.4)(x)

  x = layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = layers.Activation('relu')(x)

  if flag_BN:
    x = layers.BatchNormalization()(x)
  if flag_BN16:
    x = BN16.BatchNormalizationF16()(x)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = layers.Dropout(0.5)(x)

  x = layers.Flatten()(x)
  x = layers.Dense(512,kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = layers.Activation('relu')(x)
  if flag_BN:
    x = layers.BatchNormalization()(x)
  if flag_BN16:
    x = BN16.BatchNormalizationF16()(x)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(classes)(x)
  x = layers.Activation('softmax')(x)

  model = tf.keras.models.Model(img_input, x, name='vgg16')

  return model