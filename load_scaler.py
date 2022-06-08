import numpy as np
import joblib

# load file scaled nya
sc = joblib.load('sc.joblib')
# varibel sc ada scaler nya

# data nya begini

# product_score : 3.7
# qty : 16.0
# freight_price : 15.628125
# product_weight_g : 850.0
# lag_price : 35.0
# comp_1 : 103.2333333
# ps1 : 4.1
# fp1 : 22.3
# comp_2 : 35.0
# ps2 : 3.7
# fp2 : 15.628125

# unit_price : 35.0

# trus diubah kebentuk numpy array kaya gini

nilai = np.array([[  3.7      ,  16.       ,  15.628125 ,  35.       , 850.       ,
         35.       , 103.2333333,   4.1      ,  22.3      ,  35.       ,
          3.7      ,  15.628125 ]])

# Setelah itu di scale variabel nilai

hasil_scale = sc.transform(nilai)

# hasil scale tersimpan pada variabel hasil_scale