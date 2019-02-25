import os
import cv2
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt

DATA_DIR = "../datasets/dogs_vs_cats/"
TRAIN_DIR = '../datasets/dogs_vs_cats_data/train'
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50
TRAIN_NUMPY_FILE = 'train_data.npy'

CAT_DIR = "../datasets/dogs_vs_cats/Cat"
DOG_DIR = "../datasets/dogs_vs_cats/Dog"

training_data = []
# train_x = []  # features
# train_y = []  # labels


def convert_data():
    training_data = []
    dog_count = 0
    cat_count = 0

    for img in os.listdir(TRAIN_DIR):
        label = img.split('.')[-3]
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path)
        # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # training_data.append([np.array(img), np.array(label)])
        if label == 'cat':
            cat_count += 1
            cv2.imwrite(os.path.join(CAT_DIR, str(cat_count) + ".jpg"), img)
        elif label == 'dog':
            dog_count += 1
            cv2.imwrite(os.path.join(DOG_DIR, str(dog_count) + ".jpg"), img)

    return training_data


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([np.array(img_array), np.array(class_num)])
                print(class_num)
            except Exception as e:
                pass

    random.shuffle(training_data)
    # np.save(file=TRAIN_NUMPY_FILE, arr=training_data)
    print(len(training_data))

    x = []
    y = []

    for features, labels in training_data:
        x.append(features)
        y.append(labels)

    train_x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    # train_y = np.array(y).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    pickle_out = open("train_x.pickle", "wb")
    pickle.dump(train_x, pickle_out)
    pickle_out.close()

    pickle_out = open("train_y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


def load_training_data():
    # training_data = np.load(TRAIN_NUMPY_FILE)
    pickle_in = open("train_x.pickle", "rb")
    train_x = pickle.load(pickle_in)

    pickle_in = open("train_y.pickle", "rb")
    train_y = pickle.load(pickle_in)

    print(train_x[1])


create_training_data()
# load_training_data()
