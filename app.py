import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model/model_rf_clf_rev.pkl", "rb"))


@flask_app.route("/")
def Home():
    return render_template("home.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    output = {0: "No", 1: "Yes"}

    return render_template(
        "predict.html",
        prediction_text="{}".format(
            output[prediction[0]]))


@flask_app.route('/data-set')
def dataSet():
    return render_template('bank_dataset_filt.html')


@flask_app.route('/predict')
def predicts():
    return render_template('predict.html')


if __name__ == "__main__":
    flask_app.run(debug=True)