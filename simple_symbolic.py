from sklearn.linear_model import LinearRegression
import pred_actual

def lin_reg(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)


    #get equation and features
    coefs = list(model.coef_)
    intercept = model.intercept_
    features = x_train.columns

    #create equation string
    equation = ""
    for i in range(len(coefs)):
        if coefs[i] > 0:
            equation += " + "+str(coefs[i])+"*"+features[i]
        else:
            equation += " - "+str(abs(coefs[i]))+"*"+features[i]
    if intercept > 0:
        equation += " + " + str(intercept)
    else:
        equation += " - " + str(abs(intercept))
    if equation[:3] == " + ":
        equation = equation[3:]
    equation = y_train.name + " = " + equation

    image = pred_actual.graph(y_train, y_pred)
    
    #metrics
    r_score, mse = pred_actual.metrics(y_train, y_pred)

    return equation, r_score, mse, image