# Load libraries
import flask
import pandas as pd
import tensorflow as tf
import keras as K
from keras.models import load_model
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

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        x=pd.DataFrame.from_dict(params, orient='index').transpose()
        y = x[["product_score", "qty", "freight_price", 
            "product_weight_g", "lag_price", "comp_1", "ps1", "fp1", "comp_2", "ps2","fp2"]]
        # with graph.as_default():
        nilai = np.array([[  3.7      ,  16.0       ,  15.628125 ,  35.0       , 850.0       ,
         35.0       , 103.2333333,   4.1      ,  22.3      ,  35.0       ,
          3.7      ,  15.628125 ,   0       ,   0       ,   0       ,
          0       ,   1       ,   0       ,   0       ,   0      ,
          0      ]])
        sc = joblib.load('ml_models/repfit_scaler.joblib')

        prediction_result = str(model.predict(x)[0][0])
        hasil_scale = sc.transform(nilai)
        
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
