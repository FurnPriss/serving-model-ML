#! /bin/bash
curl -X POST -H "Content-Type: application/json" -d @input_data_json/input_prepredict_object.json http://127.0.0.1:5000/prepredict
