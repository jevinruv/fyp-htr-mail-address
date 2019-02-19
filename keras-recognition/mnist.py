import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class MNISTReader:
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    model = None
    predict_model = None
    MODEL_NAME = 'mnist_num_reader.model'

    test = None

    def init(self):
        mnist = tf.keras.datasets.mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        self.test = self.x_test
        # to set the values between 0 and 1
        self.x_train = tf.keras.utils.normalize(self.x_train, axis=1).reshape(self.x_train.shape[0], -1)
        self.x_test = tf.keras.utils.normalize(self.x_test, axis=1).reshape(self.x_test.shape[0], -1)

    def create_model(self):
        self.model = tf.keras.models.Sequential()  # feed forward (most common)
        # model.add(tf.keras.layers.Flatten())   #Flatten the images! Could be done with numpy reshape
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=self.x_train.shape[1:]))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # 10 because dataset is numbers from 0 - 9

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, epochs=3)

        val_loss, val_acc = self.model.evaluate(self.x_test, self.y_test)
        print(val_acc, val_loss)

        tf.keras.models.save_model(model=self.model, filepath=self.MODEL_NAME)
        # self.model.save_model(model=self.model, filepath='mnist_num_reader.model')

    def load_model(self):
        self.predict_model = tf.keras.models.load_model(filepath=self.MODEL_NAME)

    def do_predict(self):
        predictons = self.predict_model.predict([self.x_test])
        # print(predictons)
        print(np.argmax(predictons[4]))
        plt.imshow(self.test[4], cmap=plt.cm.binary)
        plt.show()

    def main(self):
        self.init()
        # self.create_model()
        self.load_model()
        self.do_predict()


mnist_reader = MNISTReader()
mnist_reader.main()

# print(x_train[0])
# plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.show()
