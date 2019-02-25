import tensorflow as tf


class Model:
    # constant declaration
    BATCH_SIZE = 50
    IMG_SIZE = (128, 32)
    TEXT_LENGTH = 32

    def __init__(self, char_list):
        self.char_list = char_list
        self.model_id = 0

        # cnn layer
        self.input_img = tf.placeholder(dtype=tf.float32, shape=(self.BATCH_SIZE, self.IMG_SIZE[0], self.IMG_SIZE[1]))
        cnn_out = self.build_cnn(self.input_img)  # get 4D cnn array

        # rnn layer
        rnn_out = self.build_rnn(cnn_out)  # get 3D rnn array

        # ctc layer
        (self.loss, self.decoder) = self.build_ctc(rnn_out)

        self.optimizer = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)

        (self.sess, self.saver) = self.prepare_TF()

    def build_cnn(self, input):
        "create a conv net with 5 layers"

        # convert to 4 dimensional array -  [no_samples, height, width, colour_channels]
        cnn_in = tf.expand_dims(input=input, axis=3)

        feature_values = [1, 32, 64, 128, 128, 256]
        fixed_values = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)] # used for stride
        no_layers = len(fixed_values)

        pool = cnn_in

        # create conv net 5 layers
        for i in range(no_layers):
            kernel = tf.Variable(tf.truncated_normal([5, 5, feature_values[i], feature_values[i + 1]], stddev=0.1))
            conv = tf.nn.conv2d(input=pool, filter=kernel, padding='SAME', strides=(1, 1, 1, 1))
            relu = tf.nn.relu(conv)
            pool = tf.nn.max_pool(relu,
                                  ksize=(1, fixed_values[i][0], fixed_values[i][1], 1),
                                  strides=(1, fixed_values[i][0], fixed_values[i][1], 1),
                                  padding='VALID')
        return pool

    def build_rnn(self, cnn_input):
        rnn_in = tf.squeeze(cnn_input, axis=[2])
        no_hidden = 256
        n_layers = 2

        cells = [tf.contrib.rnn.LSTMCell(num_units=no_hidden, state_is_tuple=True) for _ in range(n_layers)]
        stacked = tf.contrib.rnn.MultiRNNCell()

    # def build_ctc(self, rnn_input):

    # def prepare_TF(self):
