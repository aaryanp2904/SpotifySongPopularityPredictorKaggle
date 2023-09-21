from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

X_train = pd.read_csv("../../Datasets/Processed/ModelSelection/Train/XTrain.csv")
X_test = pd.read_csv("../../Datasets/Processed/ModelSelection/Test/XTest.csv")

y_train = pd.read_csv("../../Datasets/Processed/ModelSelection/Train/yTrain.csv").values.ravel()
y_test = pd.read_csv("../../Datasets/Processed/ModelSelection/Test/yTest.csv").values.ravel()

model = Pipeline([('polynom', PolynomialFeatures(degree=3)),
                  ("linear", LinearRegression(fit_intercept=False))])
model.fit(X_train, y_train)

pred = model.predict(X_test)

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