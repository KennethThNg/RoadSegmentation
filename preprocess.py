import numpy as np
import scipy


def process(train, ground_truth=None, gt=True):
    '''
    Process raw data
    :param train: raw training data
    :param ground_truth: raw ground truth corresponding data
    :param gt: true if ground truth is given
    :return: processed raw data
    '''
    X_train = (train-0.5)/0.5
    if gt:
        Y_ground_truth = (ground_truth - ground_truth.min())/(ground_truth.max()-ground_truth.min())
        return X_train, Y_ground_truth
    else:
        return X_train

def data_augmentation(train, ground_truth):
    '''
    Augmentation of data set of 6 (6 flips are provided)
    :param train: processed train
    :param ground_truth: processed corresponding ground truth
    :return: augmented data, list
    '''
    imgs_processed = []
    labels_processed = []

    for i in range(len(train)):
        img = train[i]
        img_gt = ground_truth[i]

        img_ud = np.flipud(img)
        img_gt_ud = np.flipud(img_gt)

        img_lr = np.fliplr(img)
        img_gt_lr = np.fliplr(img_gt)

        img_180 = np.rot90(img, 1)
        img_gt_180 = np.rot90(img_gt, 1)

        angle = np.random.randint(180)
        img_angle1 = scipy.ndimage.interpolation.rotate(img, angle, mode='reflect', reshape=False, order=0)
        img_gt_angle1 = scipy.ndimage.interpolation.rotate(img_gt, angle, mode='reflect', reshape=False, order=0)

        angle = np.random.randint(180)
        img_angle2 = scipy.ndimage.interpolation.rotate(img, angle, mode='reflect', reshape=False, order=0)
        img_gt_angle2 = scipy.ndimage.interpolation.rotate(img_gt, angle, mode='reflect', reshape=False, order=0)

        angle = np.random.randint(180)
        img_angle3 = scipy.ndimage.interpolation.rotate(img, angle, mode='reflect', reshape=False, order=0)
        img_gt_angle3 = scipy.ndimage.interpolation.rotate(img_gt, angle, mode='reflect', reshape=False, order=0)

        imgs_processed.extend([img, img_ud, img_lr, img_180, img_angle1, img_angle2, img_angle3])
        labels_processed.extend(
            [img_gt, img_gt_ud, img_gt_lr, img_gt_180, img_gt_angle1, img_gt_angle2, img_gt_angle3])

    return imgs_processed, labels_processed