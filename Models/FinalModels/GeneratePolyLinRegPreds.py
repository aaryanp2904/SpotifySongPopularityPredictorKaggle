import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def poly_lin_reg_predictions(x_train, y_train, x_test, variance_mean, prediction_path):
    print("------------------------------------------------------------------")
    print("Creating Polynomial Linear Regression pipeline...")
    model = Pipeline([('polynom', PolynomialFeatures(degree=3)),
                      ("linear", LinearRegression(fit_intercept=False))])

    print("Fitting training data to PLR model...")
    model.fit(x_train, y_train)

    ids = x_test["id"]
    x_test = x_test.drop(["id"], axis=1)

    print("Generating predictions...")
    predictions = model.predict(x_test)

    res = []

    print("Scaling up predictions...")
    for item in predictions:
        res.append(min(100, max((item * variance_mean["std"]) + variance_mean["mean"], 0)))

    result = pd.DataFrame({"id": ids, "popularity": res})

    print("Saving scaled predictions to csv...")
    result.to_csv(prediction_path + "PolyLinRegPredictions.csv", index=False)
