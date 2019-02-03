import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

# print(mnist.train.images)

single_image = mnist.train.images[1].reshape(28, 28)    #get the first image from the array and resize it to 28x28
plt.imshow(single_image, cmap="gray")   #grayscale and plot the image
plt.show()

print("min", single_image.min(), " max", single_image.max())
