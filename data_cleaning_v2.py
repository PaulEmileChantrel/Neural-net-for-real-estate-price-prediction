#Data cleaning for US real estate
import requests
import urllib.parse
import pandas as pd
import geopandas
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from nn_utils_function import *
from ast import literal_eval as make_tuple
#df = pd.read_csv('realtor-data.csv')

# We only keep NYC
# df1 = df[df['city']=='New York City']
# df1['city_code'] = 0
# df2 = df[df['city']=='New York']
# df2['city_code'] = 1
# df3 = df[df['city']=='Brooklyn']
# df3['city_code'] = 2
# df4 = df[df['city']=='Manhattan']
# df4['city_code'] = 3
# df5 = df[df['city']=='Bronx']
# df5['city_code'] = 4
# df6 = df[df['city']=='Jersey City']
# df6['city_code'] = 5
# df = pd.concat([df1,df2,df3,df4,df5,df6],ignore_index=True)
# print(df.shape)

# #We remove row with empty value
# df = df.fillna('')
# df = df[df['bath']!='']
# df = df[df['bed']!='']
# df = df[df['house_size']!='']
# df = df[df['street']!='']
# df.drop_duplicates(ignore_index=True,inplace=True,subset=['full_address'])
# #If we don't have a acre lot data but we have a house size, we consider that the acre lot is the house size
# sub_df = df[df['acre_lot']=='']
# df = df[df['acre_lot']!='']
# sub_df['acre_lot'] = sub_df['house_size']/43560
#
# df = pd.concat([df,sub_df],ignore_index=True)
#
# df.to_csv('realtor_data_wo_lat_long.csv', index=False)
# print(df.shape)
# # Step 1 : Convert the addresses into latitude and longitude
# locator = Nominatim(user_agent="myGeocoder")
# # 1 - conveneint function to delay between geocoding calls
# geocode = RateLimiter(locator.geocode, min_delay_seconds=0.2)
# # 2- - create location column
# df['location'] = df['full_address'].apply(geocode)
# df.to_csv('realtor_data_w_lat_long.csv', index=False)
# df = pd.read_csv('realtor_data_w_lat_long.csv')
# df = df.fillna('')
# df = df[df['point']!='']
#
# # # 3 - create longitude, laatitude and altitude from location column (returns tuple)
# # df['point'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
# # # 4 - split point column into latitude, longitude and altitude columns
# arr_of_tuple = []
# for i in range(df.shape[0]):
#     l = make_tuple(df['point'][i])
#     arr_of_tuple.append(l)
#
# df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(arr_of_tuple, index=df.index)
# print(df)
# df.to_csv('realtor_data_w_lat_long.csv', index=False)
#
# #Step 2 : clean the data -> only keep row with, long and lat and delete useless columns
#
df = pd.read_csv('realtor_data_w_lat_long.csv')
# df = df.fillna('')
# df = df[df['latitude']!='']


#df.drop(columns=['Unnamed: 0','point','altitude','lat','lon','is_appt'],inplace=True)
df.drop(columns=['point','altitude'],inplace=True)
#df.to_csv('cleaned_realtor_data_w_lat_long.csv', index=False)



df['r'],df['theta']= carth_to_polar(df['latitude'],df['longitude'],x0= 40.720127,y0=-73.990247)
df = df[df['r']<1]
df.to_csv('cleaned_realtor_data_w_lat_long.csv', index=False)
df = pd.read_csv('cleaned_realtor_data_w_lat_long.csv')
print(df.shape)
#Step 3: plot data
import folium
import webbrowser
import os
import numpy as np
import matplotlib.pyplot as plt

lat = df['latitude'][0]
long = df['longitude'][0]

map1 = folium.Map(
    location=[lat,long],
    tiles='cartodbpositron',
    zoom_start=8,
)
# print(df['price'].max(),df['price'].min())
# ax = df['price'].plot.hist(bins=30)
# plt.yscale('log')
# plt.xlabel('Price')
# plt.title('Price Distribution of New York properties')
# plt.show()

# df['rooms'] = df['bathrooms']+df['bedrooms']
#
# x =  df['rooms'].to_numpy()
# y = df['price'].to_numpy()
# plt.scatter(x,y,marker='+')
# plt.ylabel('Price')
# plt.yscale('log')
# plt.xlabel('Rooms (Bedrooms + Bathrooms)')
# plt.title('Real Estate price as a function of the rooms number')
# plt.show()
#
# r,theta = carth_to_polar(df['latitude'],df['longitude'],x0= 40.720127,y0=-73.990247)
#
# x = r.to_numpy()
# y = df['price'].to_numpy()
# plt.scatter(x,y,marker='+')
# plt.ylabel('Price')
# plt.yscale('log')
# plt.xlabel('Distance ftom Downtown Manhattan')
# plt.title('Real Estate price as a function of the distance from Downtown Manhattan')
# plt.show()

def color_producer(col):

    max_price = df['price'].max()
    min_price = df['price'].min()

    sep1 = (max_price - min_price)/32
    sep2 = (max_price - min_price)/16
    sep3 = (max_price - min_price)/8
    sep4 = (max_price - min_price)/4

    if col>sep4:
        return 'red'
    elif col > sep3:
        return 'orange'
    elif col > sep2:
        return 'yellow'
    elif col > sep1:
        return 'purple'
    else:
        return 'blue'

df.apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude"]],fill=True, # Set fill to True
                            fill_color=color_producer(row["price"]),
                            color = color_producer(row["price"]),
                            fill_opacity=1).add_to(map1), axis=1)
map1.save("map.html")
webbrowser.open('file://' + os.path.realpath("map.html"))
