import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

n_features = 10
n_dense_neurons = 3

x = tf.placeholder(tf.float32, (None, n_features))  # inputs

W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))  # weights

b = tf.Variable(tf.ones([n_dense_neurons]))  # bias

xW = tf.matmul(x, W)  # matrix multiplication
z = tf.add(xW, b)
a = tf.sigmoid(z)  # activation function

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    layer_out = sess.run(a, feed_dict={x: np.random.random([1, n_features])})
    print(layer_out)


"Simple Regression Example"

x_data = np.linspace(0,10,10) + np.random.uniform(-1.5, 1.5, 10)
x_data

y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
y_label

plt.plot(x_data, y_label, "*")
plt.show()  #display graph