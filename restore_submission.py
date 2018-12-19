import numpy as np
import tensorflow as tf
from helpers import *

def submission_per_patch(session, graph, images, img_number, window_size, patch_size=16, stride=16, threshold=0.5):
    '''
    :param session: Give a tensorflow session to be run on.
    :param graph: Give a default graph
    :param images: test images
    :param img_number: number of test images to be predicted
    :param window_size: window size (be consistent with the graph)
    :param patch_size: patch size
    :param stride: stride
    :param threshold: threshold at which the prediction is put to 1 (0.5 since balanced)
    :return: yield prediction per patch of size patch, per image
    '''
    padding = (window_size - patch_size) // 2

    # Access feed_dict and output
    X = graph.get_tensor_by_name("X:0")
    p = graph.get_tensor_by_name("p:0")
    op_to_restore = graph.get_tensor_by_name("out:0")

    # Processed and crop images
    test_images = process(images, gt=False)
    test_patches = np.array(img_crop(test_images[img_number - 1], patch_size, patch_size, stride, padding))

    # Test augmentation
    flip_ud = np.array([np.flipud(test_patches[i]) for i in range(test_patches.shape[0])])
    flip_lr = np.array([np.fliplr(test_patches[i]) for i in range(test_patches.shape[0])])
    flip_rot90 = np.array([np.rot90(test_patches[i], 1) for i in range(test_patches.shape[0])])
    flip_rot180 = np.array([np.rot90(test_patches[i], 2) for i in range(test_patches.shape[0])])
    flip_rot270 = np.array([np.rot90(test_patches[i], 3) for i in range(test_patches.shape[0])])

    # Run each augmentation
    Z = session.run(op_to_restore, feed_dict={X: test_patches, p: 1})
    Z_ud = session.run(op_to_restore, feed_dict={X: flip_ud, p: 1})
    Z_lr = session.run(op_to_restore, feed_dict={X: flip_lr, p: 1})
    Z_rot90 = session.run(op_to_restore, feed_dict={X: flip_rot90, p: 1})
    Z_rot180 = session.run(op_to_restore, feed_dict={X: flip_rot180, p: 1})
    Z_rot270 = session.run(op_to_restore, feed_dict={X: flip_rot270, p: 1})

    # Raw predictions
    pred_fold = np.array([sigmoid(Z[i]) for i in range(test_patches.shape[0])])
    pred_fold_ud = np.array([sigmoid(Z_ud[i]) for i in range(test_patches.shape[0])])
    pred_fold_lr = np.array([sigmoid(Z_lr[i]) for i in range(test_patches.shape[0])])
    pred_fold_rot90 = np.array([sigmoid(Z_rot90[i]) for i in range(test_patches.shape[0])])
    pred_fold_rot180 = np.array([sigmoid(Z_rot180[i]) for i in range(test_patches.shape[0])])
    pred_fold_rot270 = np.array([sigmoid(Z_rot270[i]) for i in range(test_patches.shape[0])])

    # Average predictions to 0-1 by thresholding
    pred_mean = (pred_fold + pred_fold_ud + pred_fold_lr + pred_fold_rot90 + pred_fold_rot180 + pred_fold_rot270) / 6
    prediction = (pred_mean > threshold) * 1


    nb = 0
    print("Processing " + str(img_number - 1))
    for j in range(0, images[img_number - 1].shape[1], patch_size):
        for i in range(0, images[img_number - 1].shape[0], patch_size):
            label = int(prediction[nb])
            nb += 1
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


def tf_restore_predict(filename_saver, images, window_size, to_submit_filename=None, threshold=0.5):
    '''
    Provide a submission csv
    :param filename_saver: path for session access
    :param images: test images
    :param window_size: window size
    :param to_submit_filename: csv filename for submission
    :param threshold: threshold to predict 1
    :return: submission.csv
    '''
    # Restore meta graph and weights
    sess = tf.Session()
    saver = tf.train.import_meta_graph('../models/' + filename_saver + '.ckpt.meta')
    saver.restore(sess, '../models/' + filename_saver + '.ckpt')
    graph = tf.get_default_graph()

    # File submissions sample
    with open(to_submit_filename, 'w') as f:
        f.write('id,prediction\n')
        for nb_test in range(1, TEST_SIZE + 1, 1):
            print(nb_test)
            f.writelines('{}\n'.format(s) for s in submission_per_patch(sess, graph, images, nb_test, window_size, threshold=threshold))
