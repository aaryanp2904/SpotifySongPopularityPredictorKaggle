import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras import layers
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import numpy as np

###############################     Reading Initial Data    ###################################

# Read in data
X_train = pd.read_csv("../../Datasets/Processed/ModelSelection/Train/XTrain.csv")
X_test = pd.read_csv("../../Datasets/Processed/ModelSelection/Test/XTest.csv")

y_train = pd.read_csv("../../Datasets/Processed/ModelSelection/Train/yTrain.csv")
y_test = pd.read_csv("../../Datasets/Processed/ModelSelection/Test/yTest.csv")

########################### Training ############################################


# Create an early stopper object
early_stopper = EarlyStopping(patience=4)

# Create our base neural network
model = keras.Sequential()

model.add(layers.Input(shape=(15,)))
model.add(layers.Dense(1024, activation="relu"))
model.add(layers.Dense(512, activation="sigmoid"))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(128, activation="sigmoid"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, verbose=1,
                    batch_size=32, callbacks=[early_stopper])

model.save("model1.h5")

preds = model.predict(X_test)

print(preds)
print(np.sqrt(mean_squared_error(preds, y_test)))