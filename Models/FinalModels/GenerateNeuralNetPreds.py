import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras import layers
from keras.models import load_model
import math

currModel = 2

currPro = f"../../Datasets/Processed/FinalDatasets" \
          f"/PreprocessingModel{currModel}/"

predPath = "../../Predictions/"

test = pd.read_csv(currPro + "test.csv")

model = load_model(f"NeuralNetworkModelFiles/model{currModel}.h5")

ids = test["id"]

varMean = {}

with open(currPro + "varMean.txt", "r") as f:
    data = f.readline()
    vals = data.split(", ")
    varMean["std"] = math.sqrt(float(vals[0][1:-1]))
    varMean["mean"] = float(vals[1][1:-1])

test = test.drop(["id"], axis=1)

preds = model.predict(test)

res = []

for item in preds:
    res.append(min(100, max((item[0] * varMean["std"]) + varMean["mean"], 0)))

print(preds)

result = pd.DataFrame({"id": ids, "popularity": res})

if currModel == 1:
    predAdd = "PreprocessingModel1/"
else:
    predAdd = "PreprocessingModel2/"

result.to_csv(predPath + predAdd + "NeuralNetPredictions.csv", index=False)