import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras import layers
from keras.callbacks import EarlyStopping

prepro1 = "C:/Users/aarya/PycharmProjects/SpotifyPopularityPrediction/Datasets/Processed/FinalDatasets" \
          "/PreprocessingModel1/ "
prepro2 = "C:/Users/aarya/PycharmProjects/SpotifyPopularityPrediction/Datasets/Processed/FinalDatasets" \
          "/PreprocessingModel2/ "

# Read in data
X = pd.read_csv(prepro1 + "X.csv")
y = pd.read_csv(prepro1 + "y.csv")

########################### Training ############################################


# Create an early stopper object
early_stopper = EarlyStopping(patience=4)

# Create our base neural network
model = keras.Sequential()

model.add(layers.Input(shape=(15,)))
model.add(layers.Dense(1024, activation="relu"))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

history = model.fit(X, y, epochs=20, validation_split=0.2, verbose=1,
                    batch_size=32, callbacks=[early_stopper])

model.save("./NeuralNetworkModelFiles/model3.h5")