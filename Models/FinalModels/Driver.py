from GenerateNeuralNetPreds import neural_net_predictions
from GenerateXGBoostPreds import xgb_predictions
from GeneratePolyLinRegPreds import poly_lin_reg_predictions
from GenerateAverageModelPreds import average_predictions
import math
import pandas as pd

currModel = 2


def driver():

    print(f"Using Preprocessing Model {currModel}")

    curr_model_path = f"../../Datasets/Processed/FinalDatasets" \
                      f"/PreprocessingModel{currModel}/"

    x_train, y_train = get_train_data(curr_model_path)

    x_test = get_test_data(curr_model_path)

    variance_mean = get_var_mean(curr_model_path)

    prediction_path = f"../../Predictions/PreprocessingModel{currModel}"

    neural_net_predictions(currModel, x_test, variance_mean, prediction_path)
    xgb_predictions(x_train, y_train, x_test, variance_mean, prediction_path)
    poly_lin_reg_predictions(x_train, y_train, x_test, variance_mean, prediction_path)
    average_predictions(prediction_path)


def get_train_data(curr_model_path):
    print("Getting training data...")
    x_train = pd.read_csv(curr_model_path + "X.csv")
    y_train = pd.read_csv(curr_model_path + "y.csv").values.ravel()

    return x_train, y_train


def get_test_data(curr_model_path):
    print("Getting test data...")
    return pd.read_csv(curr_model_path + "test.csv")


def get_var_mean(curr_model_path):
    print("Getting variance and mean data...")
    variance_mean = {}

    with open(curr_model_path + "varMean.txt", "r") as f:
        data = f.readline()
        vals = data.split(", ")
        variance_mean["std"] = math.sqrt(float(vals[0][1:-1]))
        variance_mean["mean"] = float(vals[1][1:-1])

    return variance_mean


if __name__ == "__main__":
    driver()
