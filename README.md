# CS-433 Machine Learning - Project 2


For the second project in the CS-433 Machine Learning course we choose to do the Road Segmentation task, with the objective of classifying aerial images and determine whether the pixels contain a road segment or background. The project was run like a CrowdAI Competition.

The content of this project is composed of the following parts:


- the folder `data` contains all the data needed for the project. It is composed of three files:

   
- `sample-submission.csv`: this shows the format in which you have to present your predictions.
   
- `test.csv`: the data set used to make predictions from the model.
   
- `train.csv`: the data set used for training the model.


- the folder `report` contains one item:

    
- `report.pdf`: the report in PDF format.
 
- the folder `notebooks` contains two items:

    
- `run_submission.ipynb` notebook that permit to run the run py-file.


- `cross_validation.ipynb` notebook that provides a cross validation demonstration.


- the folder `src` contains several py-files:

   
- **`costs.py`**: contains all the necessary functions to compute the loss of different models.
      
- `compute_mse`: computes the cost using MSE.

- `compute_logistic_loss`: computes the cost for logistic regression.      
- `compute_neg_log_likelihood`: computes the cost by negative log likelihood.

 
  
- **`cross_validation.py`**: contains all the necessary functions to perfom a cross validation.
      
- `cross_validation_per_fold`: performs cross validation based on given model.
      
- `cross_validation_per_masks`: call 'cross_validation_per_fold' with ranges of parameters to get the best combination.
      
- `split`: builds k indices for k-fold cross validation.

   

- **`features_eng.py`**: contains features engineering functions used.
        
- `fit_degree_expansion`: creates the polynomial expansion for a set of features.
      
- `fit_invert_log`: creates inverse log values of features of positive one and creates polynomial expansion for those new features.

        
- `transform_invert_log`: creates inverse log values for given columns features.

   

- **`complementaries_functions.py`**: contains complementary functions.
  
- `sigmoid`: sigmoid function.
    
- `batch_iter`: generates a minibatch iterator for a dataset.
      
- `init_w`: initializes the weight vector around 0.


- **`preprocess.py`**: contains complementary functions.
  
- `get_missing_masks`: get masks for splitting columns.
    
- `fit_remove_constant_col`: removes constant columns of the train set.
 
- `transform_remove_constant_col`: removes columns of the test set which are assumed to be constant.
      
- `fit_standardize`: standardizes columns.

- `transform_standardize`: standardizes columns with given fit mean and std.


- **`proj1_helpers.py`**: predefined help functions for the project slightly modified.
      
- `load_csv_data`: loads the data from CSV.
      
- `predict_labels`: predicts label based on data matrix and weight vector compatible with Kaggle.
      
- `create_csv_submission`: creates CSV file for submission.

   


- **`implementations.py`**: contains all regression methods used for this project.
      
- `least_squares_GD`: linear regression using gradient descent.
      
- `least_squares_SGD`: linear regression using stochastic gradient descent.
      
- `least_squares`: least squares regression using normal equations.
      
- `ridge_regression`: ridge regression using normal equations.
      
- `logistic_regression`: logistic regression using gradient descent.
      
- `reg_logistic_regression_SGD`: regularized logistic regression using gradient descent.

   

- **`run.py`**: contains the procedure that generates the exact CSV file submitted on Kaggle.
   

