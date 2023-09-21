# Spotify Song Popularity Predictor

## Directory Structure

- **Datasets** - Contains the `csv` files and data values
    - **Original** - Contains the original test and train `.csv` files provided
    - **Processed** - Contains preprocessed data for multiple purposes
      - **ModelSelection** - This directory has data used for experimentation and finding loss values by dividing the original `train.csv` file into test and train sets
        - **Test** - Test data to evaluate models
        - **Train** - Train data to fit models to
      - **FinalDatasets** - Contains X and y `.csv` files for each preprocessing model, the variance and mean of the train data and a standardized and processed version of the original `test.csv` file
        - **PreprocessingModel1** - This is the first preprocessing model and has a different algorithm for accounting for artists
        - **PreprocessingModel2** - This is the second preprocessing model and has a different algorithm for accounting for artists
- **Models** - The `.py` files that include all the code for the different models
  - **Experimenting** - These are the models that the `ModelSelection/` data was used on to test the accuracy and loss of the models
  - **FinalModels** - Contains the models and code to generate predictions based on different model architectures
    - **NeuralNetworkModelFiles** - Contains the `.h5` files generated from `NeuralNetConstructor.py`  
- **Predictions** - Contains the `.csv` files for each model's prediction, split according to which Preprocessing model's data I used
  - **PreprocessingModel1** - Predictions generated using data from the first preprocessing model
  - **PreprocessingModel2** - Predictions generated using data from the second preprocessing model
- **Preprocessing** - Contains the `.py` files which do all the preprocessing on the original `train.csv` files, including standardizing numeric features and converting categorical features such as artists and year to suitable numeric equivalents
