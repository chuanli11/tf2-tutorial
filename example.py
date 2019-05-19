
import  os
import  tensorflow as tf
from    tensorflow import  keras
from    tensorflow.keras import datasets, layers, optimizers, models, regularizers
import  argparse
import  numpy as np



# from network import VGG16

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
parser = argparse.ArgumentParser()



parser.add_argument('--train_dir', type=str, default='/tmp/cifar10_train',
                           help="Directory where to write event logs and checkpoint.")
parser.add_argument('--max_steps', type=int, default=1000000,
                            help="""Number of batches to run.""")
parser.add_argument('--log_device_placement', action='store_true',
                            help="Whether to log device placement.")
parser.add_argument('--log_frequency', type=int, default=10,
                            help="How often to log results to the console.")
parser.add_argument('--num_gpus', type=int, default=1,
                            help="How many GPUs to use.")
parser.add_argument('--bs_per_gpu', type=int, default=256,
                            help="Batch size on each GPU.")
parser.add_argument('--num_epochs', type=int, default=3,
                            help="Number of training epochs.")
parser.add_argument('--num_train_samples', type=int, default=40000,
                            help="Number of training samples.")

args = parser.parse_args()


def VGG16(input_shape):
  # Do not use subclass for easier save/load model and print summary
  weight_decay = 0.000
  num_classes = 10

  model = models.Sequential()

  model.add(layers.Conv2D(64, (3, 3), padding='same',
                   input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())

  model.add(layers.MaxPooling2D(pool_size=(2, 2)))

  model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())

  model.add(layers.MaxPooling2D(pool_size=(2, 2)))

  model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())

  model.add(layers.MaxPooling2D(pool_size=(2, 2)))


  model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())

  model.add(layers.MaxPooling2D(pool_size=(2, 2)))


  model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())

  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Dropout(0.5))

  model.add(layers.Flatten())
  model.add(layers.Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())

  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(num_classes))
  model.add(layers.Activation('softmax'))  

  return model

def normalize(X_train, X_test):
    # this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
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


def main():

    tf.random.set_seed(22)

    print('loading data...')
    (x,y), (x_test, y_test) = datasets.cifar10.load_data()
    x, x_test = normalize(x, x_test)

    x_val = x[args.num_train_samples:, :]
    y_val = y[args.num_train_samples:, :]

    x = x[:args.num_train_samples, :]
    y = y[:args.num_train_samples, :]

    train_loader = tf.data.Dataset.from_tensor_slices((x,y))
    val_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_loader = train_loader.map(prepare_cifar).shuffle(args.num_train_samples).batch(args.bs_per_gpu * args.num_gpus)
    val_loader = val_loader.map(prepare_cifar).batch(args.bs_per_gpu * args.num_gpus)
    test_loader = test_loader.map(prepare_cifar).batch(args.bs_per_gpu * args.num_gpus)      


    if args.num_gpus == 1:
        model = VGG16([32, 32, 3])
        model.compile(
                  optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])   

    else:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = VGG16([32, 32, 3])
            model.compile(
                      optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])   

    model.fit(train_loader,
              epochs=args.num_epochs,
              validation_data=val_loader,
              validation_freq=1)
    model.evaluate(test_loader)

    # Save & load weights
    # Cannot save model configuration: http://ashokrahulgade.com/coding/keras/Module1.html
    # Save weights to disk
    model.save('model.h5')
    new_model = keras.models.load_model('model.h5')     
    new_model.evaluate(test_loader)


if __name__ == '__main__':
    main()