# CS-433 Machine Learning - Project 2


For the second project in the CS-433 Machine Learning course we choose to do the Road Segmentation task, with the objective of classifying aerial images and determine whether the pixels contain a road segment or background. The project was run like a CrowdAI Competition.


CrowdAI submission username: BriceRepond
CrowdAI submission ID: TBD
Score for the model given in this submission: 0.818 

## Libraries
TensorFlow version 1.12.0

## How to run the model

The model can be run by running all the cells in the jupyter notebook called: `run.ipynb`, located in the **src**-folder.

As we had problems with restoring the pre-trained model giving the score of 0.818, the code is set to train the model from scratch, by setting the parameter _RESTORE_MODEL_ in the **parameters.py** to FALSE.


## File description
The content of this project is composed of the following parts:

- the folder `data` that should contain all the data needed for the project. It is composed of two subfolders:
   
   - `test_set_images`: the data set used to make predictions from the model, provided by the CrowdAI dataset in **test_set_images_zip**
   
   - `training`: the data set used for training the model. It is splitted into two subfolders:
      - `groundtruth`: the groundtruth images provided by the CrowdAI dataset in **training_zip**
      - `images`: the satelite training images provided by the CrowdAI dataset in **training_zip**.

The training and test data are available for download at the CrowdAI-platform at this ([Link](https://www.crowdai.org/challenges/epfl-ml-road-segmentation/dataset_files "Link")) and we therefore chose not to upload it as part of the submission.

- the folder `report` contains one item:
    
- `report.pdf`: the report in PDF format.
 
- the folder `models` contains one subfolder:
   - the folder `10_epochs_54_window_size` that contains the weights and related meta data of the pre-trained model that was created when running the model that provided the results giving the 0.818 score on the CrowdAI-platform. Due to problems with trying to restore this model, these files are not used.  

- the folder `src` that contains several py-files and one ipnyb-file:

   - **`helpers.py`**
      - `sigmoid`: performs the sigmoid function.
      - `extract_data`: Extract the images into a 4D tensor. 
      - `extract_test`: Extract images for test.
      - `img_crop_gt`: Crop an image into patches (this method is intended for ground truth images).
      - `crop_and_padding`: Pad and crop images.
      - `balanced_data`: Balanced data to make sure to have same amount of roads and non roads.  
      Functions for the CNN model:
      - `conv2d`
      - `pooling`
      - 'dense'
      - 'flattening'
      - 'activation'
      - 'optimizer_choice'
      
    - **`model.py`**
      - `Class CNN`: The CNN class.
         - `_init_`: Constructs the CNN class.
         - `model`: Provide the graph for the CNN.
         - `prepare_batches`: Creates a list with the batches.
         - `train`: Train a CNN instance
    - **`parameters.py`** 
      - Contains the parameters used to produce the CrowdAI score of 0.818.
    - **`preprocess.py`**     
      - `process`: Process raw data.
      - `data_augmentation`: Augmentation of the data set by performing six flips.
    - **`run.ipynb`** notebook that permit to run the CNN model.
      - Also contains ALL the content from the **model.py** and the **helpers.py**. This is to make sure that the CNN-model runs from scratch when running all the cells in the notebook. When trying to 1) first restore the model (as mentioned earlier), and 2) When 1) did not work, run the CNN model from scartch, we encounter import problems between the files. For this reason we choose to be completely sure of the model running by copying the content from the two abovementioned files.

 

