from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import backend as K
import tensorflow as tf

import BN16

K.set_floatx('float16')
K.set_epsilon(1e-4)

batch_size = 128
num_classes = 10
epochs = 12

BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-3

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
    bn_axis = 1
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    bn_axis = 3

x_train = x_train.astype('float16')
x_test = x_test.astype('float16')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():

  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=input_shape))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  # model.add(BatchNormalization(axis=bn_axis, name='bn_conv1',
  #                                          momentum=BATCH_NORM_DECAY,
  #                                          epsilon=BATCH_NORM_EPSILON))
  # model.add(BatchNormalization(axis=bn_axis, name='bn_conv1'))  
  model.add(BN16.BatchNormalizationF16(axis=bn_axis, name='bn_conv1',
                                           momentum=BATCH_NORM_DECAY,
                                           epsilon=BATCH_NORM_EPSILON))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))


  # model.add(BatchNormalization(axis=bn_axis, name='bn_conv1',
  #                                          momentum=BATCH_NORM_DECAY,
  #                                          epsilon=BATCH_NORM_EPSILON,
  #                                          input_shape=input_shape))

  # model.add(BN16.BatchNormalizationF16(axis=bn_axis, name='bn_conv1',
  #                                          momentum=BATCH_NORM_DECAY,
  #                                          epsilon=BATCH_NORM_EPSILON,
  #                                          input_shape=input_shape))

  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax', use_bias=False))
  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])