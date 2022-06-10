# third parties
from keras.models import load_model
import flask
import joblib
import keras as K
import pandas as pd
import tensorflow as tf

# local libraries
import utils
from utils.modelConstants import ModelConstants
from utils.repfitModel import RepfitModel

# instantiate flask
app = flask.Flask(__name__)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# we need to redefine our metric function in order
# to use it when loading the model


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.backend.get_session().run(tf.local_variables_initializer())
    return auc


# load the model, and pass in the custom metric function
global graph
graph = tf.compat.v1.get_default_graph()
model = load_model("ml_models/model_repfit_v1.h5", custom_objects={"rmse": rmse})
model.load_weights("ml_models/model_repfit_v1_weights.h5")


@app.route("/")
def hello():
    return flask.jsonify({
        "success": True,
        "message": "Hello world!"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    try:
        params = flask.request.json

        scaler_input = {
            "categories": params["categories"],
            "qty": params["qty"],
            "product_weight": params["product_weight"],
            "comp_1": params["comp_1"],
            "ps1": params["ps1"],
        }
        scaler_input = ModelConstants.fillWithDataConstants(**scaler_input)
        scaler_result = RepfitModel.scaleData(**scaler_input)

        data['unit_price'] = RepfitModel.predictPrice(scaler_result['result'])
        data["success"] = True
    except Exception as e:
        data["message"] = str(e)
        data["cause"] = str(e.__class__)

    return flask.jsonify(data)


@app.route("/prepredict", methods=["GET", "POST"])
def prepredict():
    data = {"success": False}

    try:
        params = flask.request.json
        if(params is None):
            params = flask.request.args

        if(params is not None):
            scaler_input = [list(params.values())]
            scaler = joblib.load("ml_models/scaler.joblib")
            scaler_result = scaler.transform(scaler_input)

            utils.printScalerLogs(
                scaler_input=scaler_input, scaler_result=scaler_result
            )

            scaler_input_object = dict(
                zip(utils.scaler_data_index, scaler_input[0]))
            scaler_result_object = dict(
                zip(utils.scaler_data_index, scaler_result[0]))

            data["success"] = True
            data["scaler_input"] = scaler_input_object
            data["scaler_result"] = scaler_result_object
        else:
            data["message"] = "You need to send parameters to this endpoint."
    except Exception as e:
        data["message"] = str(e)
        data["cause"] = str(e.__class__)

    return flask.jsonify(data)


@app.route("/predict_test", methods=["GET", "POST"])
def predict_test():
    data = {"success": False}

    params = flask.request.json
    if (params is None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params is not None):
        x = pd.DataFrame.from_dict(params, orient="index").transpose()

        prediction_result = str(model.predict(x)[0][0])

        # this is used to get the price from prediction result (unscaled)
        # unscaled = scaled * (max - min) + min ......(1)
        # which is in this case
        ## unit_price = prediction_result * (max - min) + min
        # max and min values are the constants
        # max and min values are the constants
        data["unit_price"] = (float(prediction_result) * (364.900000 - 19.900000) + 19.900000)
        data["success"] = True

    # return a response in json format
    return flask.jsonify(data)


@app.route("/dicttoarray", methods=["POST"])
def dicttoarray():
    data = {"success": False}

    try:
        params = flask.request.json

        # x = pd.DataFrame.from_dict(params, orient="index").transpose()
        x = list(params.values())

        data["success"] = True
        print(x)
        data["x"] = x

    except Exception as e:
        data["message"] = str(e)
        data["cause"] = str(e.__class__)

    return flask.jsonify(data)


# start the flask app, allow remote connections
app.run(host="0.0.0.0")
