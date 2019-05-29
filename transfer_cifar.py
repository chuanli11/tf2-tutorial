import tensorflow as tf

import resnet2




HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 50000

NUM_GPUS = 1
BS_PER_GPU = 128
NUM_EPOCHS = 60

BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 30), (0.01, 45)]

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

train_loader = tf.data.Dataset.from_tensor_slices((x,y))
test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

tf.random.set_seed(22)
train_loader = train_loader.map(augmentation).map(preprocess).shuffle(NUM_TRAIN_SAMPLES).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)
test_loader = test_loader.map(preprocess).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)

opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

input_shape = (32, 32, 3)
img_input = tf.keras.layers.Input(shape=input_shape)

model = resnet2.resnet56(img_input=img_input, classes=NUM_CLASSES)

model.load_weights('model.h5')

# model.compile(
#           optimizer=opt,
#           loss='sparse_categorical_crossentropy',
#           metrics=['accuracy'])
# model.evaluate(test_loader)


model.trainable = False

y = model.layers[-2].output

my_output = tf.keras.layers.Dense(10, activation='softmax',
                            kernel_initializer='he_normal',
                            kernel_regularizer=
                            tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                            bias_regularizer=
                            tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                            name='fc10')(y)


mymodel = tf.keras.models.Model(img_input, my_output, name='my')
mymodel.summary()


mymodel.compile(
          optimizer=opt,
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])
mymodel.fit(train_loader,
          epochs=NUM_EPOCHS,
          validation_data=test_loader,
          validation_freq=1)