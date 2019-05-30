import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras import backend as K

import resnet
import vgg
import BN16

import numpy as np

K.set_floatx('float16')
K.set_epsilon(1e-4)


HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 50000

NUM_GPUS = 2
BS_PER_GPU = 128
NUM_EPOCHS = 1

BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 30), (0.01, 45)]

def preprocess(x, y):
  x = tf.image.per_image_standardization(x)
  # x = tf.cast(x, "float16")
  return x, y


def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(
        x, HEIGHT + 8, WIDTH + 8)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
    x = tf.image.random_flip_left_right(x)
    return x, y	


def schedule(epoch):
  initial_learning_rate = BASE_LEARNING_RATE * BS_PER_GPU / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate


(x,y), (x_test, y_test) = keras.datasets.cifar10.load_data()

x = x.astype('float16')
x_test = x_test.astype('float16')

train_loader = tf.data.Dataset.from_tensor_slices((x,y))
test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

tf.random.set_seed(22)
train_loader = train_loader.map(augmentation).map(preprocess).shuffle(NUM_TRAIN_SAMPLES).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)
test_loader = test_loader.map(preprocess).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)

input_shape = (32, 32, 3)
img_input = tf.keras.layers.Input(shape=input_shape)
opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

if NUM_GPUS == 1:
    # model = resnet.resnetsmall(img_input=img_input, classes=NUM_CLASSES)
    model = vgg.VGG16(img_input=img_input, classes=NUM_CLASSES)
    model.compile(
              optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
else:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
	    model = resnet.resnetsmall(img_input=img_input, classes=NUM_CLASSES)
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
          epochs=NUM_EPOCHS,
          validation_data=train_loader,
          validation_freq=1,
          callbacks=[tensorboard_callback, lr_schedule_callback])
results = model.predict(train_loader)

class_id = np.argmax(results, axis=1)

print(class_id)
print(class_id.shape)
# print(results)
# print(results.shape)
# model.evaluate(test_loader)


# model.save('fp16model.h5')

# new_mymodel = tf.keras.models.load_model('fp16model.h5')
 
# new_mymodel.evaluate(test_loader)    


# tf.keras.experimental.export_saved_model(model, 'fp16model.h5')
# new_model = tf.keras.experimental.load_from_saved_model('fp16model.h5', custom_objects={'BatchNormalizationF16':BN16.BatchNormalizationF16})
# new_model.compile(
#           optimizer=opt,
#           loss='sparse_categorical_crossentropy',
#           metrics=['accuracy'])
# new_model.evaluate(test_loader)