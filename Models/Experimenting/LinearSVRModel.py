from sklearn.svm import SVR
import pandas as pd
from sklearn.metrics import mean_squared_error

# Read in data
X_train = pd.read_csv("../../Datasets/Processed/ModelSelection/Train/XTrain.csv")
X_test = pd.read_csv("../../Datasets/Processed/ModelSelection/Test/XTest.csv")

y_train = pd.read_csv("../../Datasets/Processed/ModelSelection/Train/yTrain.csv")
y_test = pd.read_csv("../../Datasets/Processed/ModelSelection/Test/yTest.csv")

svr = SVR()

svr.fit(X_train, y_train.values.ravel())

pred = svr.predict(X_test)

preds = []

for item in pred:
    preds.append(item * (477.18835399 ** (1/2)) + 25.88818139)

test = []

for item in y_test:
    test.append(item * (477.18835399 ** (1/2)) + 25.88818139)

# RMSE Computation
rmse = mean_squared_error(test, preds, squared=False)
print("RMSE : % f" % (rmse))
