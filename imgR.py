
import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard
import time
import pickle

NAME = "Hand_Gestures_convNetwork {}".format(int(time.time()))

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

X = X / 255.0

layer_sizes = [128]
conv_layers = [1]
model = Sequential()

for layer_size in layer_sizes:
    for conv_layer in conv_layers:
        NAME = "{}-conv-{}-nodes-{}".format(conv_layer, layer_size, int(time.time()))
        print(NAME)

        model.add(Conv2D(layer_size, (4 ,4), input_shape=X.shape[1:]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        for l in range(conv_layer - 1):
            model.add(Conv2D(layer_size, (4, 4)))
            model.add(Activation('relu'))
            
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(5))
        model.add(Activation('sigmoid'))

        tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'],
                      )
        model.summary()

        model.fit(X, y,
                  batch_size=64,
                  epochs=1,
                  validation_split=0.45,
                  callbacks=[tensorboard])
        
model.save('HandGestures-CNN2.model')
