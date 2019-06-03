import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

import resnet

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 50000

NUM_GPUS = 1
BS_PER_GPU = 128
NUM_EPOCHS = 3

BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 1), (0.01, 2)]

L2_WEIGHT_DECAY = 2e-4


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
  initial_learning_rate = BASE_LEARNING_RATE * BS_PER_GPU / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate


(x,y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x = x[0:NUM_TRAIN_SAMPLES, :]
y = y[0:NUM_TRAIN_SAMPLES]
x_test = x_test[0:NUM_TRAIN_SAMPLES, :]
y_test = y_test[0:NUM_TRAIN_SAMPLES]

train_loader = tf.data.Dataset.from_tensor_slices((x,y))
test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

tf.random.set_seed(22)
train_loader = train_loader.map(augmentation).map(preprocess).shuffle(NUM_TRAIN_SAMPLES).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)
test_loader = test_loader.map(preprocess).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)


input_shape = (HEIGHT, WIDTH, NUM_CHANNELS)
img_input = tf.keras.layers.Input(shape=input_shape)
opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)


if NUM_GPUS == 1:
  model = resnet.resnet56(img_input=img_input, classes=NUM_CLASSES)
  model.load_weights('model.h5')
  model.trainable = False
  feat = model.layers[-2].output
 
  if False:
    # Test the accuracy of restored model
    model.compile(
              optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.evaluate(test_loader)

  my_output = tf.keras.layers.Dense(10, activation='softmax',
                              kernel_initializer='he_normal',
                              kernel_regularizer=
                              tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                              bias_regularizer=
                              tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                              name='fc10')(feat)
  mymodel = tf.keras.models.Model(img_input, my_output, name='my')
  mymodel.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
else:
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = resnet.resnet56(img_input=img_input, classes=NUM_CLASSES)
    model.load_weights('model.h5')
    model.trainable = False
    feat = model.layers[-2].output

    if False:
      # Test the accuracy of restored model
      model.compile(
                optimizer=opt,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
      model.evaluate(test_loader)

    my_output = tf.keras.layers.Dense(10, activation='softmax',
                                kernel_initializer='he_normal',
                                kernel_regularizer=
                                tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                bias_regularizer=
                                tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                name='fc10')(feat)
    mymodel = tf.keras.models.Model(img_input, my_output, name='my')
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

lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(schedule)

mymodel.fit(train_loader,
          epochs=NUM_EPOCHS,
          validation_data=test_loader,
          validation_freq=1,
          callbacks=[tensorboard_callback, lr_schedule_callback])
mymodel.evaluate(test_loader)

mymodel.save('mymodel.h5')

new_mymodel = tf.keras.models.load_model('mymodel.h5')
 
new_mymodel.evaluate(test_loader)    
