from keras.models import load_model
import keras
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

X_train = pd.read_csv("../../Datasets/Processed/ModelSelection/Train/XTrain.csv")
X_test = pd.read_csv("../../Datasets/Processed/ModelSelection/Test/XTest.csv")

y_train = pd.read_csv("../../Datasets/Processed/ModelSelection/Train/yTrain.csv").values.ravel()
y_test = pd.read_csv("../../Datasets/Processed/ModelSelection/Test/yTest.csv").values.ravel()

model = load_model("model.h5")

pred = model.predict(X_test)

preds = []

for item in pred:
    preds.append(item * (477.18835399 ** (1/2)) + 25.88818139)

test = []

for item in y_test:
    test.append(item * (477.18835399 ** (1/2)) + 25.88818139)

# RMSE Computation
rmse = np.sqrt(mean_squared_error(test, preds))
print("RMSE : % f" % (rmse))