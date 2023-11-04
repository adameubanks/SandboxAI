import xgboost as xgb
import pred_actual

def xgb_model(X, y):
    reg = xgb.XGBRegressor()
    reg.fit(X, y)
    y_pred = reg.predict(X)

    image = pred_actual.graph(y, y_pred)

    r_score, mse = pred_actual.metrics(y, y_pred)

    reg.save_model("xgboost_model.json")

    return r_score, mse, image
