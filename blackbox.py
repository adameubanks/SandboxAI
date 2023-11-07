from sklearn.model_selection import train_test_split
import xgboost as xgb
import pred_actual

def xgb_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    reg = xgb.XGBRegressor()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    image = pred_actual.graph(y_test, y_pred)

    r_score, mse = pred_actual.metrics(y_test, y_pred)

    reg.save_model("xgboost_model.json")

    return r_score, mse, image
