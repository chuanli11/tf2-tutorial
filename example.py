import  os
import  tensorflow as tf
from    tensorflow import  keras
from    tensorflow.keras import datasets, layers, optimizers
import  argparse
import  numpy as np



from network import VGG16

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

args = parser.parse_args()

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

    train_loader = tf.data.Dataset.from_tensor_slices((x,y))
    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_loader = train_loader.map(prepare_cifar).shuffle(50000).batch(args.bs_per_gpu * args.num_gpus)
    test_loader = test_loader.map(prepare_cifar).shuffle(10000).batch(args.bs_per_gpu * args.num_gpus)      


    if args.num_gpus == 1:
        model = VGG16([32, 32, 3])
        model.compile(
                  optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])   

    else:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = VGG16([32, 32, 3])
            model.compile(
                      optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])   

    model.fit(train_loader, epochs=args.num_epochs)
    model.evaluate(train_loader)


if __name__ == '__main__':
    main()