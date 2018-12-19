import os
from helpers import *
from model import *
from parameters import *
from restore_submission import *
from preprocess import *

# Dir for train
data_dir = "../data/training/"
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/'


# Dir for test
root_dir = "../data/test_set_images/"

print('Read raw data...')
# Read raw training images
train_raw = extract_data(train_data_filename, TRAINING_SIZE)
ground_truth_raw = extract_data(train_labels_filename, TRAINING_SIZE)

print('Read test images...')
# Get filenames and images for all the 50 submission images
image_dir = [root_dir + "test_{}/".format(i) for i in range(1, TEST_SIZE+1)]
filenames = [fn for imdir in image_dir for fn in os.listdir(imdir)]
images = np.asarray([extract_test(image_dir[i-1] + filenames[i-1]) for i in range(1, TEST_SIZE+1)])

print('Processed raw data...')
# Processed data
train, ground_truth = process(train_raw, ground_truth_raw)

print('Apply data augmentation...')
# Data augmentation
augmented_train, augmented_ground_truth = data_augmentation(train, ground_truth)

# Initialize CNN
if not RESTORE_MODEL:
    print('Compute CNN...')
    cnn = CNN(window_size=WINDOW_SIZE)
    cnn.model()
    cnn.train(augmented_train, augmented_ground_truth,
                                             saver_filename=SAVER_FILENAME, number_epochs=NUM_EPOCHS)

print('Restore model and predict for submission...')
tf_restore_predict(SAVER_FILENAME, images, WINDOW_SIZE, 'submission_sample.csv')