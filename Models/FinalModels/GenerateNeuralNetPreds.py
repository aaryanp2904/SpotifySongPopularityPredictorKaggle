import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import load_model

currModel = 2


def neural_net_predictions(curr_model, x_test, variance_mean, prediction_path):
    print("------------------------------------------------------------------\nLoading neural network...")

    model = load_model(f"NeuralNetworkModelFiles/model{curr_model}.h5")

    ids = x_test["id"]

    x_test = x_test.drop(["id"], axis=1)

    print("Generating predictions...")
    predictions = model.predict(x_test)

    res = []

    print("Scaling up predictions...")
    for item in predictions:
        res.append(min(100, max((item[0] * variance_mean["std"]) + variance_mean["mean"], 0)))

    result = pd.DataFrame({"id": ids, "popularity": res})

    print("Saving scaled predictions to csv...")
    result.to_csv(prediction_path + "NeuralNetPredictions.csv", index=False)
    
