import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

class CNNModel:
    train_x = None
    train_y = None
    IMG_SIZE = 50

    def load_data(self):
        self.train_x = pickle.load(open("train_x.pickle", "rb"))
        self.train_y = pickle.load(open("train_y.pickle", "rb"))

        self.train_x = self.train_x.reshape(self.train_x.shape[0], -1)
        self.train_y = self.train_x.reshape(self.train_x.shape[0], -1)

        self.train_x = self.train_x / 255.0

    def create_model(self):
        model = Sequential()

        # input layer
        model.add(Conv2D(64, (3, 3), input_shape=self.train_x.shape[1:]))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))

        # output layer
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(self.train_x, self.train_y, batch_size=32, epochs=3, validation_split=0.1)

    def main(self):
        self.load_data()
        self.create_model()


model = CNNModel()
model.main()
