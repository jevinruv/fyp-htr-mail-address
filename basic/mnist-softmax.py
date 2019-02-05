import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

# print(mnist.train.images)

single_image = mnist.train.images[1].reshape(28, 28)  # get the first image from the array and resize it to 28x28
plt.imshow(single_image, cmap="gray")  # grayscale and plot the image
plt.show()

# print("min", single_image.min(), " max", single_image.max())


# PLACEHOLDERS
x = tf.placeholder(tf.float32, shape=[None, 784])

# VARIABLES
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# CREATE GRAPH OPERATIONS
y = tf.matmul(x, W) + b

# LOSS FUNCTION
y_true = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

# OPTIMIZER
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

# CREATE SESSION
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x:batch_x, y:batch_y})

    # EVALUATE THE MODEL
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
