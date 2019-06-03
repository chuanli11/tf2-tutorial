import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import applications
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


import resnet

HEIGHT = 224
WIDTH = 224
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 5000

NUM_GPUS = 2
BS_PER_GPU = 4
NUM_EPOCHS = 20

BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 10), (0.01, 15)]
L2_WEIGHT_DECAY = 2e-4

MEAN = [103.939, 116.779, 123.68]

def preprocess(x, y):
  # x = preprocess_input(x)
  x = tf.compat.v1.image.resize_images(x, (HEIGHT, WIDTH))
  x = tf.cast(x, tf.float32)
  x = x - MEAN
  return x, y


def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(
        x, HEIGHT + 32, WIDTH + 32)
    # x = tf.compat.v1.image.resize_images(x, (HEIGHT + 32, WIDTH + 32))
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


x = x[0:NUM_TRAIN_SAMPLES, :]
y = y[0:NUM_TRAIN_SAMPLES]
x_test = x_test[0:NUM_TRAIN_SAMPLES, :]
y_test = y_test[0:NUM_TRAIN_SAMPLES]

# print(x_test.shape)
# print(y_test.shape)
# import sys
# sys.exit()
train_loader = tf.data.Dataset.from_tensor_slices((x,y))
test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

tf.random.set_seed(22)
train_loader = train_loader.map(preprocess).map(augmentation).shuffle(NUM_TRAIN_SAMPLES).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)
test_loader = test_loader.map(preprocess).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)


input_shape = (HEIGHT, WIDTH, 3)
img_input = tf.keras.layers.Input(shape=input_shape)
opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

if NUM_GPUS == 1:
    model = applications.ResNet50(weights = "imagenet", include_top=False, input_shape = (WIDTH, HEIGHT, NUM_CHANNELS))
    # model.trainable = False
    x = model.output
    my_output = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    my_output = tf.keras.layers.Dense(10, activation='softmax',
                                kernel_initializer='he_normal',
                                kernel_regularizer=
                                tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                bias_regularizer=
                                tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                name='fc10')(my_output)
    mymodel = tf.keras.models.Model(model.input, my_output, name='my')
    mymodel.compile(
              optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])    
else:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():  
      model = applications.ResNet50(weights = "imagenet", include_top=False, input_shape = (WIDTH, HEIGHT, NUM_CHANNELS))
      model.trainable = False
      x = model.output
      my_output = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
      my_output = tf.keras.layers.Dense(10, activation='softmax',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=
                                  tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                  bias_regularizer=
                                  tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                  name='fc10')(my_output)
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

lr_schedule_callback = keras.callbacks.LearningRateScheduler(schedule)

mymodel.fit(train_loader,
          epochs=NUM_EPOCHS,
          validation_data=test_loader,
          validation_freq=1,
          callbacks=[tensorboard_callback, lr_schedule_callback])
mymodel.evaluate(test_loader)