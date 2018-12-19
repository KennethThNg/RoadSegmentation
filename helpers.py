import math
import os
import tensorflow as tf
import matplotlib.image as mpimg



def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def extract_data(filename, num_images):
    '''
    Extract the images into a 4D tensor [image index, y, x, channels].
    Values are scaled from [0, 1] instead of [0,255].
    :param filename: image filename.
    :param num_images: number of images.
    :return: arrays of images.
    '''

    imgs = []
    for i in range(1, num_images +1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')


    return np.asarray(imgs)


def extract_test(image_filename):
    '''
    Extract images for test.
    :param image_filename: images files
    :return: arrays of images.
    '''

    if os.path.isfile(image_filename):
        print ('Loading ' + image_filename)
        return mpimg.imread(image_filename)


def img_crop_gt(im, w, h, stride):
    '''
    Crop an image into patches (this method is intended for ground truth images).
    :param im: image to crop
    :param w: width
    :param h: height
    :param stride: stride
    :return: cropping images
    '''

    assert len(im.shape) == 2, 'Expected greyscale image.'
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    for i in range(0, imgheight, stride):
        for j in range(0, imgwidth, stride):
            im_patch = im[j:j + w, i:i + h]
            list_patches.append(im_patch)
    return list_patches


def img_crop(im, w, h, stride, padding):
    '''
    Crop an image into patches, taking into account mirror boundary conditions.
    :param im: image to crop
    :param w: width
    :param h: height
    :param stride: stride
    :param padding: padding
    :return: cropping images
    '''

    assert len(im.shape) == 3, 'Expected RGB image.'
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    im = np.lib.pad(im, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
    for i in range(padding, imgheight + padding, stride):
        for j in range(padding, imgwidth + padding, stride):
            im_patch = im[j - padding:j + w + padding, i - padding:i + h + padding, :]
            list_patches.append(im_patch)
    return list_patches


def crop_and_padding(train, ground_truth, patch_size, stride, window_size):
    '''
    Pad and crop images
    :param train: images array
    :param ground_truth: groundtruth corresponding to train
    :param patch_size: patch size
    :param stride: stride
    :param window_size: window size
    :return: list of padding and cropping images
    '''
    padding = (window_size - patch_size) // 2

    crop_and_padding_train = []
    crop_and_padding_ground_truth = []

    for i in range(len(train)):
        cp_train = img_crop(train[i], patch_size, patch_size, stride, padding)
        cp_gt = img_crop_gt(ground_truth[i], patch_size, patch_size, stride)

        crop_and_padding_train.extend(cp_train)
        crop_and_padding_ground_truth.extend(cp_gt)

    return crop_and_padding_train, crop_and_padding_ground_truth


def balanced_data(train_data, train_gt, threshold=0.25):
    '''
    Balanced data to make sure to have same amount of roads and non roads
    :param train_data: images train
    :param train_gt: corresponding ground truth
    :param threshold: percentage of pixel to be considered as  road
    :return: sublist with balanced data
    '''
    labels = np.array([(np.mean(train_gt[i]) > threshold) * 1 for i in range(len(train_gt))])
    idx0 = [i for i, v in enumerate(labels) if v == 0]
    idx1 = [i for i, v in enumerate(labels) if v == 1]
    min_c = min(len(idx0), len(idx1))
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    train_data = [train_data[i] for i in new_indices]
    train_gt = [train_gt[i] for i in new_indices]
    labels = [(np.mean(train_gt[i]) > threshold) * 1 for i in range(len(train_gt))]
    return train_data, train_gt, labels

# All functions for the CNN model
def conv2d(layer, filters_size, kernel_size, s, padding, activation='relu'):
    if activation == 'relu':
        activation = tf.nn.relu
    return tf.layers.conv2d(layer, filters=filters_size, kernel_size=[kernel_size, kernel_size], strides=[s, s],
                            padding=padding, activation=activation)


def pooling(layer, k=2, s=2, pool_type='max'):
    if pool_type == 'max':
        return tf.layers.max_pooling2d(layer, pool_size=[k, k], strides=s)


def dense(layer, inputs_size, outputs_size, he_std=0.1):
    weights = tf.Variable(tf.truncated_normal([inputs_size, outputs_size], stddev=he_std))
    biases = tf.Variable(tf.constant(he_std, shape=[outputs_size]))
    layer = tf.matmul(layer, weights) + biases
    return layer


def flattening_layer(layer):
    # make it single dimensional
    input_size = layer.get_shape().as_list()
    new_size = input_size[-1] * input_size[-2] * input_size[-3]
    return tf.reshape(layer, [-1, new_size]), new_size


def activation(layer, activation='no_activation'):
    if activation == 'relu':
        return tf.nn.relu(layer)
    elif activation == 'soft_max':
        return tf.nn.softmax(layer)
    elif activation == 'sigmoid':
        return tf.nn.sigmoid(layer)
    else:
        return layer


def optimizer_choice(name='adam', lr=0.001):
    if name == 'GD':
        return tf.train.GradientDescentOptimizer(lr)
    elif name == 'adam':
        return tf.train.AdamOptimizer(lr)