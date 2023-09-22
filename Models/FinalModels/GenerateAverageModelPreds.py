import pandas as pd


def average_predictions(prediction_path):
    print("------------------------------------------------------------------")
    print("Reading predictions from models...")
    predictions_nn = pd.read_csv(prediction_path + "NeuralNetPredictions.csv")
    predictions_plr = pd.read_csv(prediction_path + "PolyLinRegPredictions.csv")

    # The XGB predictions tend to worsen performance so we choose not to include them
    # predsXGB = pd.read_csv(prediction_path + "XGBoostPredictions.csv")

    predictions = pd.DataFrame({"id": predictions_nn["id"], "neural_net": predictions_nn["popularity"], "plr": predictions_plr["popularity"]})

    print("Calculating mean of chosen predictions...")
    predictions["popularity"] = predictions[["neural_net", "plr"]].mean(axis=1)

    print("Saving mean predictions to csv...")
    predictions[["id", "popularity"]].to_csv(prediction_path + "AverageModelPredictions.csv", index=False)
