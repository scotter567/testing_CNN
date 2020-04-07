import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

TRAIN_X = pickle.load(open("TRAIN_X.pickle", "rb"))
TRAIN_y = pickle.load(open("TRAIN_y.pickle", "rb"))

TEST_X = pickle.load(open("TEST_X.pickle", "rb"))
TEST_y = pickle.load(open("TEST_y.pickle", "rb"))

TRAIN_X = TRAIN_X/255
TEST_X = TEST_X/255

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = TRAIN_X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(TRAIN_X, TRAIN_y, batch_size=32, epochs=100, validation_split=0.2)
results = model.evaluate(TEST_X, TEST_y)
print(results)
