import joblib
import os

model_path = "models/"
model_filename = "xgb_dropout_model.joblib"


def predict_dropout(df):
    model = joblib.load(os.path.join(model_path, model_filename))
    pred = model.predict(df)
    proba = model.predict_proba(df)[:, 1]
    return pred, proba
