# Neural Net for Toronto real estate price prediction

In this project, we want to build a neural net to predict the price of Toronto real estate.
To do this, we use the following dataset: https://www.kaggle.com/datasets/mangaljitsingh/torontoproperties


## Data Cleaning

First of all, we need to clean the data (with data_cleaning.py).
1) We add the longitude and latitude of every property
2) We drop the row where no longitude and latitude are found
3) We drop columns with no helpful data

## First look at the data

To take a first look at the data, we plot it on a map with different colours depending on the price.

<p align='center'>
<img width="724" alt="Capture d’écran 2022-11-25 à 02 25 22" src="https://user-images.githubusercontent.com/96018383/203926761-c1853666-f0c0-4d29-beb8-c9e2d8278219.png">
</p>

Here is the same data when we zoom in on Toronto Downtown :

<p align='center'>
<img width="724" alt="Capture d’écran 2022-11-25 à 02 37 02" src="https://user-images.githubusercontent.com/96018383/203926765-25389eb4-b871-4896-aaee-9e1679d2a996.png">
</p>

We can also look at the price distribution of the property (with a log scale for the y-axis):
<p align='center'>
<img src='https://user-images.githubusercontent.com/96018383/203933963-7ebcf916-7d2c-4603-a4c5-bac250367046.png'>
</p>

From this figure, we can see that most of the real estate price is between $0 and $500,000. The frequency of apparition is inverlsely correlated with the price.

We can also look a the price against the number of rooms:
<p align='center'>
  <img src='https://user-images.githubusercontent.com/96018383/203933914-72c5ffae-772d-47cd-b64a-b82d0d61a8fe.png'>
</p>

From this figure, we can see that the more rooms we have, the higher the price is. We can also notice that some properties don't have any rooms.

## Creation of new features

So far, we only have 4 features to guess the price of a property: the number of bedrooms, the number of bathrooms, the latitude and the longitude.
We create  4 more features: the number of rooms (bedrooms + bathrooms), the polar coordinates r and $\theta$ (from the latitude and longitude) with a center in downtown Toronto and we map the region to a different number.

## Neural Network

We use a neural network with 5 dense layers with 2048, 1024, 512, 256 and 128 neurons.
After training over 1000 epochs, we get the following results for the loss function.

<p align='center'>
  <img src='https://user-images.githubusercontent.com/96018383/203937654-84d3bdf0-8fe2-45ad-8c74-5d2919ef1499.png'>
 </p>
 
 After the training, we only manage to get 50% of the training set correct and 80% of the testing set. 
 (We consider a correct match if the price is correctly predicted within a +/- $50,000 range, which is a lot).
 
 From the above figure, we can see that the loss function stop improving on the testing set which means we can't improve the result by adding more epochs.
 
 The failure of predicting the real estate price correctly probably comes from the lack of enough data.
 First, we only have around 5000 properties(split between training and testing) and second, we have few data points for each property. We only have rooms information and the property location but we don't know if it's a house or an apartment, the number of floors, the sqft, whether it has a garden or not, what is the view, etc.
 
 If we try with a simple regression (with a gradient descent) we have similar results (regression.py)
 
 Conclusion: we need more and better data to improve this model.
 




