import tflearn
import cv2  # manipulate images (gray scale, resize)
import numpy as np  # manipulate arrays
import os  # for directories
import matplotlib.pyplot as plt
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tqdm import tqdm


class CatsVsDogs:
    TRAIN_DIR = '../datasets/dogs_vs_cats_data/train'
    TEST_DIR = '../datasets/dogs_vs_cats_data/test'
    IMG_SIZE = 50
    LEARNING_RATE = 0.001  # 1e-3
    MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LEARNING_RATE, '6cnn')

    model = None

    def label_img(self, img):
        word_label = img.split('.')[-3]

        #   convert to one-hot array [isCat, isDog]
        if word_label == 'cat':
            return [1, 0]
        elif word_label == 'dog':
            return [0, 1]

    def create_train_data(self):
        training_data = []

        for img in tqdm(os.listdir(self.TRAIN_DIR)):
            label = self.label_img(img)  # calls label_img() to get one-hot value
            path = os.path.join(self.TRAIN_DIR, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])

        shuffle(training_data)
        np.save('train_data.npy', training_data)
        return training_data

    def create_test_data(self):
        testing_data = []

        for img in tqdm(os.listdir(self.TEST_DIR)):
            path = os.path.join(self.TEST_DIR, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            testing_data.append([np.array(img), np.array(img_num)])

        shuffle(testing_data)
        np.save('test_data.npy', testing_data)
        return testing_data

    def conv_neural_network(self):
        cnn = input_data(shape=[None, self.IMG_SIZE, self.IMG_SIZE, 1], name='input')

        cnn = conv_2d(cnn, 32, 5, activation='relu')  # strides = amount of pixel shift (steps)
        cnn = max_pool_2d(cnn, 5)

        cnn = conv_2d(cnn, 64, 5, activation='relu')
        cnn = max_pool_2d(cnn, 5)

        cnn = conv_2d(cnn, 32, 5, activation='relu')  # strides = amount of pixel shift (steps)
        cnn = max_pool_2d(cnn, 5)

        cnn = conv_2d(cnn, 64, 5, activation='relu')
        cnn = max_pool_2d(cnn, 5)

        cnn = conv_2d(cnn, 32, 5, activation='relu')  # strides = amount of pixel shift (steps)
        cnn = max_pool_2d(cnn, 5)

        cnn = fully_connected(cnn, 1024, activation='relu')
        cnn = dropout(cnn, 0.8)

        cnn = fully_connected(cnn, 2, activation='softmax')
        cnn = regression(cnn, optimizer='adam', learning_rate=self.LEARNING_RATE, loss='categorical_crossentropy',
                         name='targets')

        self.model = tflearn.DNN(cnn, tensorboard_dir='log')

    def create_model(self, train_data, test_data):
        train = train_data[:-500]
        test = test_data[-500:]

        X_train = np.array([i[0] for i in train]).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        Y_train = [i[1] for i in train]

        x_test = np.array([i[0] for i in test]).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        y_test = [i[1] for i in train]

        self.model.fit(
            {'input': X_train}, {'targets': Y_train},
            n_epoch=3,
            validation_set=({'input': x_test}, {'targets': y_test}),
            snapshot_step=500,
            show_metric=True,
            run_id=self.MODEL_NAME)

        self.model.save(self.MODEL_NAME)

    def recognize(self):

        test_data = np.load('test_data.npy')
        fig = plt.figure()

        for num, data in enumerate(test_data[:12]):
            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(self.IMG_SIZE, self.IMG_SIZE, 1)
            model_out = self.model.predict([data])[0]

            if np.argmax(model_out) == 1:
                str_label = 'Dog'
            else:
                str_label = 'Cat'

            y.imshow(orig, cmap='gray')
            plt.title(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)

        plt.show()

    def main(self):
        if os.path.isfile('train_data.npy') and os.path.isfile('test_data.npy'):
            train_data = np.load('train_data.npy')
            test_data = np.load('test_data.npy')
            print('npy loaded!')

        else:
            train_data = self.create_train_data()
            test_data = self.create_test_data()
            print('npy created!')

        self.conv_neural_network()
        print('Check Model File')

        if not os.path.isfile('{}.meta'.format(self.MODEL_NAME)):
            print('Model Creating!')
            self.create_model(train_data, test_data)

        self.model.load(self.MODEL_NAME)
        print('Model Loaded!')
        self.recognize()


cvd = CatsVsDogs()
cvd.main()
