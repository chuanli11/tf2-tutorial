from tensorflow import keras
keras.backend.set_floatx('float16')

input_layer = keras.layers.Input(shape=(16,16,3))
x = keras.layers.BatchNormalization(axis=3)(input_layer) # <<Fails here
x = keras.layers.Conv2D(32, (3,3))(x)