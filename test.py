#http://hilite.me/

import tensorflow as tf

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

    if False:
      for xx, yy in train_loader:
        print(xx.shape)
        print(yy.shape)
        break

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

    log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir,
      update_freq=args.bs_per_gpu * args.num_gpus * 10,
      histogram_freq=1)


    model.fit(train_loader,
              epochs=args.num_epochs,
              validation_data=val_loader,
              validation_freq=1,
              callbacks=[tensorboard_callback])
    model.evaluate(test_loader)

    model.save('model.h5')

    new_model = keras.models.load_model('model.h5')
 
    new_model.evaluate(test_loader)

if __name__ == '__main__':
    main()