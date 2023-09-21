import xgboost as xgb
import pandas as pd
import math

currModel = 2

currPro = f"../../Datasets/Processed/FinalDatasets" \
          f"/PreprocessingModel{currModel}/"

predPath = "../../Predictions/"

X_train = pd.read_csv(currPro+ "X.csv")
y_train = pd.read_csv(currPro+ "y.csv").values.ravel()

X_test = pd.read_csv(currPro + "test.csv")

DM_train = xgb.DMatrix(data=X_train, label=y_train)

params={"objective":"reg:squarederror", "n_estimators":50, "learning_rate":0.215, "subsample":0.1, "booster":"gblinear", "max_leaves":700}

varMean = {}

with open(currPro + "varMean.txt", "r") as f:
    data = f.readline()
    vals = data.split(", ")
    varMean["std"] = math.sqrt(float(vals[0][1:-1]))
    varMean["mean"] = float(vals[1][1:-1])

ids = X_test["id"]
X_test = X_test.drop(["id"], axis=1)

DM_test = xgb.DMatrix(data=X_test)

xgb_r = xgb.train(params=params, dtrain=DM_train, num_boost_round=20)

preds = xgb_r.predict(DM_test)

res = []

for item in preds:
    res.append(min(100, max((item * varMean["std"]) + varMean["mean"], 0)))

print(preds)

result = pd.DataFrame({"id": ids, "popularity": res})

if currModel == 1:
    predAdd = "PreprocessingModel1/"
else:
    predAdd = "PreprocessingModel2/"

result.to_csv(predPath + predAdd + "XGBoostPredictions.csv", index=False)
