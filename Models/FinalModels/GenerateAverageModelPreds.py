import pandas as pd

currModel = 2

predPath = f"../../Predictions/PreprocessingModel{currModel}/"

predsNN = pd.read_csv(predPath + "NeuralNetPredictions.csv")
predsPLR = pd.read_csv(predPath + "PolyLinRegPredictions.csv")
predsXGB = pd.read_csv(predPath + "XGBoostPredictions.csv")

preds = pd.DataFrame({"id": predsNN["id"], "neural_net": predsNN["popularity"], "plr": predsPLR["popularity"]})

preds["popularity"] = preds[["neural_net", "plr"]].mean(axis = 1)

preds[["id", "popularity"]].to_csv(predPath + "AverageModelPredictions.csv", index=False)