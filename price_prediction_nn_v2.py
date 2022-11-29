# Price prediction with the US data
import folium
import webbrowser
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
from sklearn.model_selection import train_test_split

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from autils import *
import random
from nn_utils_function import *
# Import Data

df = pd.read_csv('realtor-data.csv')
#df_numbers = df.drop(columns=['address','location','region'])


# We only keep NYC
df1 = df[df['city']=='New York City']
df2 = df[df['city']=='New York']
df = pd.concat([df1,df2],ignore_index=True)


#We remove row with empty value
df = df.fillna('')
df = df[df['bath']!='']
df = df[df['bed']!='']
df = df[df['acre_lot']!='']
df = df[df['house_size']!='']
df.drop_duplicates(ignore_index=True,inplace=True,subset=['full_address'])
df.to_csv('new_york_real_estate.csv')
print(df.shape)
# regions = set(df['region'])
# r_dict = {}
# i = 0
# for region in regions:
#     r_dict[region]=i
#     i+=1
#
# df['region_idx'] =''
# for i in range(len(df['region'])):
#     df.at[i,'region_idx'] = r_dict[df['region'][i]]

#df['rooms'] = df['bedrooms']+df['bathrooms']


#Normalized the Data
df['bed'],mean_bedrooms,max_minus_min_bedrooms = normalize_data(df['bed'])
df['bath'],mean_bathrooms,max_minus_min_bathrooms = normalize_data(df['bath'])
df['acre_lot'],mean_acre,max_minus_min_acre = normalize_data(df['acre_lot'])
df['house_size'],mean_house,max_minus_min_house = normalize_data(df['house_size'])


df['price'] = np.log(df['price'])
df['price'],mean_price,max_minus_min_price = normalize_data(df['price'])


df.to_csv('test.csv')




#split between data (X) and label(y)
x_df = df[['bed','bath','acre_lot','house_size']]
X = x_df.to_numpy().astype('float32')
y = df['price'].to_numpy().astype('float32')




X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15, random_state=1)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
# print(X_train,y_train)


tf.random.set_seed(1234)

E_train = []
E_cv = []

model = Sequential([
    tf.keras.Input(shape=(4,)),
    #Dense(2048, activation='relu'),
    #Dense(1024, activation='relu'),
    #Dense(512, activation='relu'),
    Dense(25, activation='relu'),
    Dense(12,activation='relu'),
    Dense(1,activation='linear')

],name='re_nn_v2')

p_error = [0.05,0.10,0.15,0.2,0.3]
for p in p_error: #calcul of random error baseline
    _,_,_,E_train = calculate_errors(model,X_train,y_train,E_train,p,mean_price,max_minus_min_price)
    failed_X,failed_prediction,failed_y,E_cv = calculate_errors(model,X_test,y_test,E_cv,p,mean_price,max_minus_min_price)

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
)

history = model.fit(
    X_train,y_train,
    epochs=500,
    validation_data=(X_test,y_test)
)

model.save('nn_v2_a_0p0001')
plot_loss_tf(history)

_for p in p_error: #calcul of error
    _,_,_,E_train = calculate_errors(model,X_train,y_train,E_train,p,mean_price,max_minus_min_price)
    failed_X,failed_prediction,failed_y,E_cv = calculate_errors(model,X_test,y_test,E_cv,p,mean_price,max_minus_min_price)

plt.show()
