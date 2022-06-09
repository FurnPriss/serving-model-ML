# third parties
from keras.models import load_model
import flask
import joblib
import keras as K
import pandas as pd
import tensorflow as tf

# local libraries
import utils

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

@app.route("/prepredict_test", methods=["GET", "POST"])
def prepredict_test():
    scaler = joblib.load('ml_models/scaler.joblib')
    scaled_result = scaler.transform(utils.dummy_scaler_input)
    
    # print 
    utils.printScalerDataToConsole(utils.scaler_data_index, utils.dummy_scaler_input)
    utils.printScalerDataToConsole(utils.scaler_data_index, scaled_result)
    
    dummy_scaler_input_object = dict(zip(utils.scaler_data_index, utils.dummy_scaler_input[0]))
    scaled_result_object = dict(zip(utils.scaler_data_index, scaled_result[0]))
    return flask.jsonify({
        'success': True,
        'data': {
            'scaler_input': dummy_scaler_input_object,
            'scaler_result': scaled_result_object,
        }
    })

# define a predict function as an endpoint 
@app.route("/predict_test", methods=["GET","POST"])
def predict_test():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        x=pd.DataFrame.from_dict(params, orient='index').transpose()
        
        prediction_result = str(model.predict(x)[0][0])
        
        # this is used to get the price from prediction result (unscaled)
        ## unscaled = scaled * (max - min) + min ......(1)
        # which is in this case
        ## your_price = prediction_result * (max - min) + min
        ### max and min values are the constants
        data["your_price"] = (float(prediction_result) * (364.900000 - 19.900000) + 19.900000)
        data["success"] = True

    # return a response in json format 
    return flask.jsonify(data)

# start the flask app, allow remote connections 
app.run(host='0.0.0.0', debug=True)
