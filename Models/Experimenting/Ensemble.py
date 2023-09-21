from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from keras import layers
from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin

class XGBoostWrapper(BaseEstimator, RegressorMixin):

    def __init__(self):
        self.model = None

    def fit(self, X, y):
        DM_train = xgb.DMatrix(data=X, label=y)
        params = {"objective": "reg:squarederror", "n_estimators": 50, "learning_rate": 0.215, "subsample": 0.1,
                  "booster": "gblinear", "max_leaves": 700}
        self.model = xgb.train(params=params, dtrain=DM_train, num_boost_round=20)
        return self.model

    def predict(self, X):
        DM_pred = xgb.DMatrix(data=X)
        return self.model.predict(DM_pred)

X_train = pd.read_csv("../../Datasets/Processed/ModelSelection/Train/XTrain.csv")
X_test = pd.read_csv("../../Datasets/Processed/ModelSelection/Test/XTest.csv")

y_train = pd.read_csv("../../Datasets/Processed/ModelSelection/Train/yTrain.csv").values.ravel()
y_test = pd.read_csv("../../Datasets/Processed/ModelSelection/Test/yTest.csv").values.ravel()

params={"objective":"reg:squarederror", "n_estimators":50, "learning_rate":0.215, "subsample":0.1, "booster":"gblinear", "max_leaves":700}

# gbr = GradientBoostingRegressor(loss="squared_error")
gbr = XGBoostWrapper()
polylinreg = Pipeline([('polynom', PolynomialFeatures(degree=3)),
                  ("linear", LinearRegression(fit_intercept=False))])
svr = SVR()
nn = load_model("model.h5")

vr = VotingRegressor(estimators=[("gbr", gbr), ("plr", polylinreg)], verbose=True)

vr.fit(X_train, y_train)

pred = vr.predict(X_test)

preds = []

for item in pred:
    preds.append(item * (477.18835399 ** (1/2)) + 25.88818139)

test = []

for item in y_test:
    test.append(item * (477.18835399 ** (1/2)) + 25.88818139)

# xgb_r = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, learning_rate=0.215, subsample=0.1, booster="gblinear")

# RMSE Computation
rmse = mean_squared_error(test, preds, squared=False)
print("RMSE : % f" % (rmse))