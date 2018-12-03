from src.cnn_implementation import *
import numpy
import tensorflow as tf

class CNN:
    def __init__(self):
        print('init done')
        self.X = None
        self.y = None
        self.network = None
        self.training_op = None
        self.out = None
        self.session = None
    def model(self, conv_params, fc_params, optimizer='GD', learning_rate=0.003):

        # init the graph
        graph = tf.Graph()
        graph.seed = 1


        with graph.as_default():

            # init the placeholder (None is given for more flexibility in batch_size)
            # 16,16 will be put as parameters in a further version
            self.X = tf.placeholder(
                    tf.float32,
                    shape=[None, 16, 16, 3], name='X')

            self.y = tf.placeholder(tf.float32, shape=[None, 2], name='y')


            he_init = tf.contrib.layers.variance_scaling_initializer()

            # Initialize first convolution step
            network = conv2d(self.X,
                             conv_params['params1']['filter_size'],
                             conv_params['params1']['kernel_size'],
                             conv_params['params1']['strides'],
                             conv_params['params1']['padding'],
                             conv_params['params1']['activation'])

            network = pooling(network)

            # for loop to allow different sizes of convolotional steps
            for i in range(2,len(conv_params)+1, 1):
                conv_par = conv_params['params'+str(i)]
                print(conv_par)
                network = conv2d(network,
                             conv_par['filter_size'],
                             conv_par['kernel_size'],
                             conv_par['strides'],
                             conv_par['padding'],
                             conv_par['activation'])

                network = pooling(network)

            # flatten last convolution step for full connected NN
            network, flatten_size = flattening_layer(network)


            # Initialize first full connected step
            network = dense(network,
                         flatten_size,
                         fc_params['params1']['output_size'])

            network = activation(network)

            # for loop to allow different sizes of full connected NN
            for i in range(2,len(fc_params)+1, 1):
                fc_par = fc_params['params'+str(i)]
                print(fc_par)
                network = dense(network,
                             fc_par['input_size'],
                             fc_par['output_size'])

                network = activation(network, fc_par['activation'])

            # Outputs, probability if last activation is a softmax
            self.out = network


            # Init the loss function (in a further version we will allow
            # different losses (find the best to minimize F1-Score)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.out, labels=self.y))

            ## Initialize the choosen Optimizer
            optimizer = optimizer_choice('GD', learning_rate)
            self.training_op = optimizer.minimize(loss)

            # Initialize all Variables
            init = tf.global_variables_initializer()

        self.session = tf.Session(config=None, graph=graph)
        self.session.run(init)

    def train(self, train_data, train_labels, number_epochs=3, batchsize=16):
        train_size=train_data.shape[0]

        for iepoch in range(number_epochs):

            batchset = prepare_batches(train_size, batchsize)

            for batch_indices in batchset:
                batch_data = train_data[batch_indices, :, :, :]
                batch_labels = train_labels[batch_indices]

                feed_dict = {
                    self.X: batch_data,
                    self.y: batch_labels,
                   }
                self.session.run(self.training_op, feed_dict=feed_dict)

    def predict(self, test):
        prediction = self.session.run(self.out, feed_dict={self.X: test})
        return prediction
