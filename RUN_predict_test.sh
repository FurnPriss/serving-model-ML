#! /bin/bash
curl -X POST -H "Content-Type: application/json" -d @input_data_json/input_predict_test.json http://127.0.0.1:5000/predict_test
