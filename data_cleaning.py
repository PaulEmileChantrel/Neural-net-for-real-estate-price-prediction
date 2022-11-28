import requests
import urllib.parse
import pandas as pd
import geopandas
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from nn_utils_function import *

# df = pd.read_csv('clean_combined_toronto_property_data.csv',sep=';')
#
# # Step 1 : Convert the addresses into latitude and longitude
# locator = Nominatim(user_agent="myGeocoder")
# # 1 - conveneint function to delay between geocoding calls
# geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
# # 2- - create location column
# df['location'] = df['address'].apply(geocode)
# # 3 - create longitude, laatitude and altitude from location column (returns tuple)
# df['point'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
# # 4 - split point column into latitude, longitude and altitude columns
# df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(df['point'].tolist(), index=df.index)
#
# df.to_csv('clean_combined_toronto_property_data_with_lat_long.csv',sep=';', index=False)

#Step 2 : clean the data -> only keep row with, long and lat and delete useless columns

# df = pd.read_csv('clean_combined_toronto_property_data_with_lat_long.csv',sep=';')
# df = df.fillna('')
# df = df[df['latitude']!='']
#
#
# #df.drop(columns=['Unnamed: 0','point','altitude','lat','lon','is_appt','pricem'],inplace=True)
# df.drop(columns=['point','altitude','pricem'],inplace=True)
#
# df.to_csv('cleaned_combined_toronto_property_data_with_lat_long.csv',sep=';', index=False)

df= pd.read_csv('cleaned_combined_toronto_property_data_with_lat_long.csv',sep=';')

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
# plt.title('Price Distribution of Greatest Toronto Area property')
# plt.show()

df['rooms'] = df['bathrooms']+df['bedrooms']

x =  df['rooms'].to_numpy()
y = df['price'].to_numpy()
plt.scatter(x,y,marker='+')
plt.ylabel('Price')
plt.yscale('log')
plt.xlabel('Rooms (Bedrooms + Bathrooms)')
plt.title('Real Estate price as a function of the rooms number')
plt.show()
#
# r,theta = carth_to_polar(df['latitude'],df['longitude'])
#
# x =  r.to_numpy()
# y = df['price'].to_numpy()
# plt.scatter(x,y,marker='+')
# plt.ylabel('Price')
# plt.xlabel('Distance ftom Downtown Toronto')
# plt.title('Real Estate price as a function of the distance from Downtown Toronto')
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
#webbrowser.open('file://' + os.path.realpath("map.html"))
