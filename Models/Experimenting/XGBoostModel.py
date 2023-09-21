import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

X_train = pd.read_csv("../../Datasets/Processed/ModelSelection/Train/XTrain.csv")
X_test = pd.read_csv("../../Datasets/Processed/ModelSelection/Test/XTest.csv")

y_train = pd.read_csv("../../Datasets/Processed/ModelSelection/Train/yTrain.csv").values.ravel()
y_test = pd.read_csv("../../Datasets/Processed/ModelSelection/Test/yTest.csv").values.ravel()

DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test = xgb.DMatrix(data=X_test, label=y_test)

params={"objective":"reg:squarederror", "n_estimators":50, "learning_rate":0.215, "subsample":0.1, "booster":"gblinear", "max_leaves":700}

xgb_r = xgb.train(params=params, dtrain=DM_train, num_boost_round=20)

pred = xgb_r.predict(DM_test)

preds = []

for item in pred:
    preds.append(item * (477.18835399 ** (1/2)) + 25.88818139)

test = []

for item in y_test:
    test.append(item * (477.18835399 ** (1/2)) + 25.88818139)

# RMSE Computation
rmse = np.sqrt(mean_squared_error(test, preds))
print("RMSE : % f" % (rmse))
