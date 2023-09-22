import xgboost as xgb
import pandas as pd


def xgb_predictions(x_train, y_train, x_test, variance_mean, prediction_path):
    print("------------------------------------------------------------------")
    print("Creating XGBoost model parameters...")
    dm_train = xgb.DMatrix(data=x_train, label=y_train)

    params = {"objective": "reg:squarederror", "n_estimators": 50, "learning_rate": 0.215, "subsample": 0.1,
              "booster": "gblinear", "max_leaves": 700}

    ids = x_test["id"]

    x_test = x_test.drop(["id"], axis=1)

    dm_test = xgb.DMatrix(data=x_test)

    print("Training XGBoost model...")
    xgb_r = xgb.train(params=params, dtrain=dm_train, num_boost_round=20)

    print("Generating predictions...")
    predictions = xgb_r.predict(dm_test)

    res = []

    print("Scaling up predictions...")
    for item in predictions:
        res.append(min(100, max((item * variance_mean["std"]) + variance_mean["mean"], 0)))

    result = pd.DataFrame({"id": ids, "popularity": res})

    print("Saving scaled predictions to csv...")
    result.to_csv(prediction_path + "XGBoostPredictions.csv", index=False)
