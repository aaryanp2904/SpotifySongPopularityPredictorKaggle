import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

X = pd.read_csv("../../Datasets/Processed/FinalDatasets/PreprocessingModel1/X.csv")
y = pd.read_csv("../../Datasets/Processed/FinalDatasets/PreprocessingModel1/y.csv")

gbm_param_grid = {
    "subsample": np.arange(.05, 1, .05),
    "max_depth": np.arange(3, 20, 1),
    "colsample_bytree": np.arange(.1, 1.05, .05),
    "learning_rate": np.arange(0.05, 1.05, .05),
    "n_estimators": [50, 100, 200]
}

gbm = xgb.XGBRegressor()

randomized_mse = RandomizedSearchCV(estimator=gbm, param_distributions=gbm_param_grid, n_iter=20, scoring="neg_mean_squared_error", cv=4, verbose=1)

randomized_mse.fit(X, y)

print("Best parameters: ", randomized_mse.best_params_)
print("Lowest RMSE: ", np.sqrt(np.abs(randomized_mse.best_score_)))
