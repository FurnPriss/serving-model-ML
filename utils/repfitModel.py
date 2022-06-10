# third party libraries
from keras.models import load_model
import joblib
import keras as K
import pandas as pd


# local libraries
from utils.modelConstants import ModelConstants


class RepfitModel:
    @staticmethod
    def predictPrice(scaler_result):
        try:
            model = load_model("ml_models/model_repfit_v1.h5",
                               custom_objects={"rmse": RepfitModel.__rmse})
            model.load_weights("ml_models/model_repfit_v1_weights.h5")

            x = pd.DataFrame.from_dict(
                scaler_result, orient="index").transpose()
            prediction_result = model.predict(x)[0][0]
            # this is used to get the price from prediction result (unscaled)
            # unscaled = scaled * (max - min) + min ......(1)
            # which is in this case
            ## unit_price = prediction_result * (max - min) + min
            # max and min values are the constants
            # max and min values are the constants
            return (prediction_result * (364.000000 - 19.900000) + 19.900000)
        except Exception as e:
            raise e

    @staticmethod
    def scaleData(**params):
        input_data = [list(params.values())]

        try:
            scaler = joblib.load("ml_models/scaler.joblib")
            result_data = scaler.transform(input_data)

            input_object = dict(
                zip(ModelConstants.getModelConstants(), input_data[0]))
            result_object = dict(
                zip(ModelConstants.getModelConstants(), result_data[0]))

            return {
                "input": input_object,
                "result": result_object
            }

        except Exception as e:
            raise e

    @staticmethod
    def __rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    # we need to redefine our metric function in order
    # to use it when loading the model
