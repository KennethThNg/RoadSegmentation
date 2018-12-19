import numpy as np
import tensorflow as tf


class CNN:
    '''
    Construct the CNN class.
    '''
    def __init__(self, window_size=62, patch_size=16, stride=16, p=0.25):
        print('init done')
        self.X = None
        self.y = None
        self.p = p
        self.network = None
        self.training_op = None
        self.out = None
        self.session = None
        self.stride = stride
        self.window_size = window_size
        self.patch_size = patch_size
        self.padding = (self.window_size - self.patch_size) // 2
        self.accuracy = None
        self.loss = None

    def model(self):
        '''
        Provide the graph for the CNN.
        '''
        # init the graph
        # graph = tf.Graph()
        # graph.seed = 1

        tf.reset_default_graph()

        # with graph.as_default():

        # init the placeholder (None is given for more flexibility in batch_size)
        # 16,16 will be put as parameters in a further version
        self.X = tf.placeholder(
            tf.float32,
            shape=[None, self.window_size, self.window_size, 3], name='X')

        self.y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
        self.p = tf.placeholder(tf.float32, name='p')

        # Initialize first convolution step
        self.network = conv2d(self.X, 64, 5, 1, 'SAME', tf.nn.relu)
        self.network = pooling(self.network)
        self.network = tf.nn.dropout(self.network, self.p)

        self.network = conv2d(self.network, 128, 3, 1, 'SAME', tf.nn.relu)
        self.network = pooling(self.network)
        self.network = tf.nn.dropout(self.network, self.p)

        self.network = conv2d(self.network, 256, 3, 1, 'SAME', tf.nn.relu)
        self.network = pooling(self.network)
        self.network = tf.nn.dropout(self.network, self.p)

        self.network = conv2d(self.network, 256, 3, 1, 'SAME', tf.nn.relu)
        self.network = pooling(self.network)
        self.network = tf.nn.dropout(self.network, self.p)

        # flatten last convolution step for full connected NN
        self.network, flatten_size = flattening_layer(self.network)

        # Initialize first full connected step
        self.network = dense(self.network,
                             flatten_size,
                             1)
        # self.network = tf.nn.dropout(self.network, self.p*2, name='out')
        self.network = tf.nn.dropout(self.network, self.p)

        self.out = self.network
        self.out = tf.identity(self.out, name='out')

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.out)
        self.loss = tf.reduce_mean(cross_entropy, name='loss')

        ## Initialize the choosen Optimizer
        optimizer = optimizer_choice('adam', 0.001)
        self.training_op = optimizer.minimize(self.loss, name='training_op')

        # Initialize all Variables
        # init = tf.global_variables_initializer()

        predicted = tf.nn.sigmoid(self.out)
        correct_pred = tf.equal(tf.round(predicted), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    def prepare_batches(self, seq, step):
        n = len(seq)
        print(n)
        res = []
        for i in range(0, n, step):
            res.append(seq[i:i + step])
        return res

    def train(self, train, ground_truth, batchsize=120, number_epochs=1, saver_filename=None, to_save=True):
        '''
        Train a cnn instance
        :param train: training augmented data
        :param ground_truth: corresponding augmented ground truth
        :param batchsize: size of each batch
        :param number_epochs: number of epochs
        :param saver_filename: name to save model
        :param to_save: true if want to save model, false else
        '''
        assert saver_filename != None
        cost_history = np.empty(shape=[1], dtype=float)

        train1, ground_truth1 = crop_and_padding(train, ground_truth, self.patch_size, self.stride, self.window_size)
        # Balanced data to improve performance
        train_data, gt, labels = balanced_data(train1, ground_truth1)

        print(train_data[0].min(), train_data[2].max(), train_data[0].mean())
        print(np.array(labels).mean())

        np.random.seed(999)

        with tf.Session() as sess:
            self.session = sess
            self.session.run(tf.global_variables_initializer())

            for iepoch in range(number_epochs):

                print(iepoch)
                train_idx_shuffle = np.arange(len(train_data))
                np.random.shuffle(train_idx_shuffle)

                batches = self.prepare_batches(train_idx_shuffle, batchsize)

                for nb, idx in enumerate(batches):

                    X_batch = [train_data[j] for j in idx]
                    Y_batch = np.array([labels[j] for j in idx]).reshape(-1, 1)

                    feed_dict = {
                        self.X: X_batch,
                        self.y: Y_batch,
                        self.p: 0.25
                    }
                    self.session.run(self.training_op, feed_dict=feed_dict)
                    cost, _, acc = self.session.run([self.loss, self.training_op, self.accuracy], feed_dict={
                        self.X: X_batch, self.y: Y_batch, self.p: 1})
                    cost_history = np.append(cost_history, acc)

                    if nb % 100 == 0:
                        print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                            nb, cost, acc))
            if to_save:
                saver = tf.train.Saver()
                saver.save(self.session, '../models/' + saver_filename + '.ckpt')