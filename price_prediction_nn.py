
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

df = pd.read_csv('cleaned_combined_toronto_property_data_with_lat_long.csv',sep=';')
#df_numbers = df.drop(columns=['address','location','region'])

regions = set(df['region'])
r_dict = {}
i = 0
for region in regions:
    r_dict[region]=i
    i+=1

df['region_idx'] =''
for i in range(len(df['region'])):
    df.at[i,'region_idx'] = r_dict[df['region'][i]]
#new features creations
df['r'],df['theta'] = carth_to_polar(df['latitude'],df['longitude'])
df['rooms'] = df['bedrooms']+df['bathrooms']


#Normalized the Data
df['bedrooms'],mean_bedrooms,max_minus_min_bedrooms = normalize_data(df['bedrooms'])
df['bathrooms'],mean_bathrooms,max_minus_min_bathrooms = normalize_data(df['bathrooms'])
df['rooms'],mean_rooms,max_minus_min_rooms = normalize_data(df['rooms'])
df['region_idx'],mean_region_idx,max_minus_min_region_idx = normalize_data(df['region_idx'])


df['latitude'],mean_latitude,max_minus_min_latitude = normalize_data(df['latitude'])
df['longitude'],mean_longitude,max_minus_min_longitude = normalize_data(df['longitude'])
df['r'],mean_r,max_minus_min_r = normalize_data(df['r'])
df['theta'],mean_theta,max_minus_min_theta = normalize_data(df['theta'])


df['price'] = np.log(df['price'])
df['price'],mean_price,max_minus_min_price = normalize_data(df['price'])


df.to_csv('test.csv')




#split between data (X) and label(y)
x_df = df[['bedrooms','bathrooms','rooms','latitude','longitude','r','theta','region_idx']]
X = x_df.to_numpy().astype('float32')
y = df['price'].to_numpy().astype('float32')




X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15, random_state=1)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
# print(X_train,y_train)


tf.random.set_seed(1234)

E_train = []
E_cv = []


model = Sequential([
    tf.keras.Input(shape=(8,)),
    Dense(2048, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128,activation='relu'),
    Dense(1,activation='linear')

],name='re_nn_v1')

p_error = [0.05,0.10,0.15,0.2,0.3]
for p in p_error: #calcul of random error baseline
    _,_,_,E_train = calculate_errors(model,X_train,y_train,E_train,p,mean_price,max_minus_min_price)
    failed_X,failed_prediction,failed_y,E_cv = calculate_errors(model,X_test,y_test,E_cv,p,mean_price,max_minus_min_price)


model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
)

history = model.fit(
    X_train,y_train,
    epochs=200,
    validation_data=(X_test,y_test)
)


plot_loss_tf(history)

for p in p_error: #calcul of error
    _,_,_,E_train = calculate_errors(model,X_train,y_train,E_train,p,mean_price,max_minus_min_price)
    failed_X,failed_prediction,failed_y,E_cv = calculate_errors(model,X_test,y_test,E_cv,p,mean_price,max_minus_min_price)

plt.show()
