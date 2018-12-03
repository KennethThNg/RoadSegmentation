import numpy
import tensorflow as tf

def prepare_batches(idxsize, batchsize):
    training_indices = range(idxsize)
    perm_indices = numpy.random.permutation(training_indices)

    res = []
    for i in range(0, idxsize, batchsize):
        res.append(perm_indices[i:i+batchsize])
    return res

def conv2d(layer, filters_size, kernel_size, s, padding, activation='relu'):
    if activation=='relu':
        activation=tf.nn.relu
    return tf.layers.conv2d(layer, filters=filters_size,  kernel_size=[kernel_size, kernel_size], strides=[s, s], padding=padding, activation=activation)

def pooling(layer, k=2, s=2, pool_type='max'):

    if pool_type=='max':
        return tf.layers.max_pooling2d(layer, pool_size=[k,k], strides=s)

def dense(layer, inputs_size, outputs_size, he_std=0.1):
    weights = tf.Variable(tf.truncated_normal([inputs_size, outputs_size],stddev=he_std))
    biases = tf.Variable(tf.constant(he_std, shape=[outputs_size]))
    layer = tf.matmul(layer,weights) + biases
    return layer

def flattening_layer(layer):
        #make it single dimensional
        input_size = layer.get_shape().as_list()
        new_size = input_size[-1] * input_size[-2] * input_size[-3]
        return tf.reshape(layer, [-1, new_size]),new_size


def activation(layer, activation='relu'):
    if activation=='relu':
        return tf.nn.relu(layer)
    elif activation=='soft_max':
        return tf.nn.softmax(layer)

def optimizer_choice(name='GD', lr=0.003):
    if name=='GD':
        return tf.train.GradientDescentOptimizer(lr)
    elif name=='adam':
        return tf.train.AdamOptimizer(lr)
