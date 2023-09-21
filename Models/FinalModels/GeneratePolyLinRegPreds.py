import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import math

currModel = 2

currPro = f"../../Datasets/Processed/FinalDatasets" \
          f"/PreprocessingModel{currModel}/"

predPath = "../../Predictions/"


X_train = pd.read_csv(currPro+ "X.csv")
y_train = pd.read_csv(currPro+ "y.csv").values.ravel()

X_test = pd.read_csv(currPro + "test.csv")

model = Pipeline([('polynom', PolynomialFeatures(degree=3)),
                  ("linear", LinearRegression(fit_intercept=False))])
model.fit(X_train, y_train)

varMean = {}

with open(currPro + "varMean.txt", "r") as f:
    data = f.readline()
    vals = data.split(", ")
    varMean["std"] = math.sqrt(float(vals[0][1:-1]))
    varMean["mean"] = float(vals[1][1:-1])

ids = X_test["id"]
X = X_test.drop(["id"], axis=1)

preds = model.predict(X)

res = []

for item in preds:
    res.append(min(100, max((item * varMean["std"]) + varMean["mean"], 0)))

print(preds)

result = pd.DataFrame({"id": ids, "popularity": res})

if currModel == 1:
    predAdd = "PreprocessingModel1/"
else:
    predAdd = "PreprocessingModel2/"

result.to_csv(predPath + predAdd + "PolyLinRegPredictions.csv", index=False)
