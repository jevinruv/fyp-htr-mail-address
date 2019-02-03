import tensorflow as tf

hello = tf.constant(2)
work = tf.constant(3)

with tf.Session() as sess:
    res = sess.run(hello*work)

print(res + 88)

