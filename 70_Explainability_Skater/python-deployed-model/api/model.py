from flask import Flask, jsonify, request
from sklearn.externals import joblib
import numpy as np
import os

MODELPATH = '../models/model.pkl'
TRAINPATH = '../bin/train.py'

def json_to_model_input(request_body):
    json_ = request_body.json
    input = json_['input']
    query = np.array(input)
    if len(query.shape) == 1:
        query = query[:, np.newaxis].T
    return query

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Note:
    # This is not a scalable solution
    # Refer to DataScience Inc.'s product offering for a scalable solution
    query = json_to_model_input(request)
    prediction = estimator.predict(query)
    return jsonify({'predictions': list(prediction)})

def train():
    from sklearn.datasets import load_boston
    from sklearn.linear_model import LinearRegression
    from sklearn.externals import joblib

    boston = load_boston()
    X = boston.data
    y = boston.target

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, MODELPATH)


if __name__ == '__main__':
    if not os.path.exists(MODELPATH):
        train()
    try:
        estimator = joblib.load(MODELPATH)
    except IOError:
        train()
        estimator = joblib.load(MODELPATH)

    app.run("0.0.0.0", debug=True)
