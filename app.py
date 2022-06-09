# Load libraries
import flask
import pandas as pd
import tensorflow as tf
import keras as K
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from pickle import load
import numpy as np
import joblib

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
model = load_model('ml_models/model_repfit_v1.h5', custom_objects={'rmse': rmse})
model.load_weights('ml_models/model_repfit_v1_weights.h5')

@app.route("/")
def hello():
    return flask.jsonify({
        "success": True,
        "message": "Hello world!"
    })

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}
    x = None

    params = flask.request.json

    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        x=pd.DataFrame.from_dict(params, orient='index').transpose()

    nilai = np.array(x.values)

    scaler_load = joblib.load('ml_models/scaler_v2.save')

    hasil_scale = scaler_load.transform(nilai)
    prediction_result = str(model.predict(hasil_scale)[0][0])
        
        # this is used to get the price from prediction result (unscaled)
        ## unscaled = scaled * (max - min) + min ......(1)
        # which is in this case
        ## your_price = prediction_result * (max - min) + min
        ### max and min values are the constants
    data["your_price"] = (float(prediction_result) * (364.900000 - 19.900000) + 19.900000)
    data["success"] = True
    data["hasil_scale"] = str(hasil_scale)

    # return a response in json format 
    return flask.jsonify(data)    

# start the flask app, allow remote connections 
app.run(host='0.0.0.0', debug=True)
