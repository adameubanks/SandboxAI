from werkzeug.utils import secure_filename
from fileinput import filename
import feature_importance
import simple_symbolic
from flask import *
import pandas as pd
import corr_matrix
import blackbox
import os

UPLOAD_FOLDER = os.path.join('static', 'uploads')
# MODEL_FOLDER = os.path.join('static', 'models')
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

try:
    file=open("key.txt","r")
    app.secret_key = file.read()
    file.close()
except FileNotFoundError:
    app.secret_key = "shhhh this key is secret or something ¯\_(ツ)_/¯ idk"

app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route('/', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
        f = request.files.get('file')
        if f:
            data_filename = secure_filename(f.filename)
            upload_folder = app.config['UPLOAD_FOLDER']
            # Ensure the directory exists
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            f.save(os.path.join(upload_folder, data_filename))
            session['uploaded_data_file_path'] = os.path.join(upload_folder, data_filename)
            return redirect(url_for("view"))
    return render_template("index.html")


@app.route('/view/')
def view():
    data_file_path = session.get('uploaded_data_file_path', None)
    uploaded_df = pd.read_csv(data_file_path, encoding='unicode_escape')
    uploaded_df = uploaded_df.dropna().select_dtypes(include=['int64', 'float64'])
    features = uploaded_df.columns
    uploaded_df_html = uploaded_df.head().to_html(classes='table-sm table-striped table-hover table-responsive')
    return render_template('view.html', dataset=uploaded_df_html, features=features)

@app.route('/feature_importance/results/', methods=['POST'])
def feature_importance_results():
    uploaded_df = pd.read_csv(session.get('uploaded_data_file_path', None), encoding='unicode_escape')
    uploaded_df = uploaded_df.dropna().select_dtypes(include=['int64', 'float64'])
    y_var = request.form.get('y_var')
    selected_features = feature_importance.k_best(uploaded_df.drop(y_var, axis=1),uploaded_df[y_var])
    return render_template('feature_importance.html', selected_features=selected_features.to_html(classes='table-sm table-striped table-hover'), y_var=y_var)

#I'm an idiot and can't find out how to pass errors in a form so these 2 functions are basically the same
@app.route('/symbolic_regression/choose_features/', methods=['POST','GET'])
def symbolic_regression_choose_features():
    uploaded_df = pd.read_csv(session.get('uploaded_data_file_path', None), encoding='unicode_escape')
    uploaded_df = uploaded_df.dropna().select_dtypes(include=['int64', 'float64'])
    uploaded_df_html = uploaded_df.head().to_html(classes='table-sm table-striped table-hover table-responsive')
    features = uploaded_df.columns
    squared_features = []
    inverse_features = []
    for feature in features:
        squared_features.append(feature+"^2")
        inverse_features.append("1/"+feature)
    return render_template("choose_features.html", dataset=uploaded_df_html, features=features, squared_features=squared_features, inverse_features=inverse_features, alert=False)

@app.route('/symbolic_regression/choose_features/?error', methods=['POST','GET'])
def symbolic_regression_choose_features_error():
    uploaded_df = pd.read_csv(session.get('uploaded_data_file_path', None), encoding='unicode_escape')
    uploaded_df = uploaded_df.dropna().select_dtypes(include=['int64', 'float64'])
    uploaded_df_html = uploaded_df.to_html(classes='table-sm table-striped table-hover table-responsive')
    features = uploaded_df.columns
    squared_features = []
    inverse_features = []
    for feature in features:
        squared_features.append(feature+"^2")
        inverse_features.append("1/"+feature)
    return render_template("choose_features.html", dataset=uploaded_df_html, features=features, squared_features=squared_features, inverse_features=inverse_features, alert=True)

@app.route('/symbolic_regression/results/', methods=['POST'])
def symbolic_regression_results():
    uploaded_df = pd.read_csv(session.get('uploaded_data_file_path', None), encoding='unicode_escape')
    uploaded_df = uploaded_df.dropna().select_dtypes(include=['int64', 'float64'])

    features = request.form.getlist('features')
    y_var = request.form.get('y_var')
    y = uploaded_df[y_var]
    for feature in features:
        if feature in uploaded_df.columns:
            continue
        elif "^2" in feature:
            uploaded_df[feature]=uploaded_df[feature[:-2]]**2
        elif "1/" in feature:
            uploaded_df[feature]=1/uploaded_df[feature[2:]]
    X = uploaded_df[uploaded_df.columns.intersection(features)].drop(y_var, axis=1, errors='ignore')
    if y_var in features:
        return redirect(url_for("symbolic_regression_choose_features_error"))
    equation, r_score, mse, image = simple_symbolic.lin_reg(X, y)
    return render_template('symbolic_regression.html', equation=equation, r_score=r_score, mse=mse, image=image, features=features)

@app.route('/blackbox/results/', methods=['POST'])
def xgboost_results():
    uploaded_df = pd.read_csv(session.get('uploaded_data_file_path', None), encoding='unicode_escape')
    uploaded_df = uploaded_df.dropna().select_dtypes(include=['int64', 'float64'])

    y_var = request.form.get('y_var')
    r2_score, mse, image = blackbox.xgb_model(uploaded_df.drop(y_var, axis=1),uploaded_df[y_var])
    return render_template('blackbox.html', r_score=r2_score, mse=mse, image=image)

@app.route('/blackbox/results/download', methods=['POST', 'GET'])
def download_xgboost():
    return send_file('xgboost_model.json', as_attachment=True)

@app.route('/correlation_matrix/results/')
def correlation_matrix():
    uploaded_df = pd.read_csv(session.get('uploaded_data_file_path', None), encoding='unicode_escape')
    uploaded_df = uploaded_df.dropna().select_dtypes(include=['int64', 'float64'])

    image = corr_matrix.graph(uploaded_df, uploaded_df.columns)

    return render_template('correlation_matrix.html', image=image)

if __name__ == '__main__':
	app.run(debug=True)
