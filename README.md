# Neural Net for real estate price prediction

In this project, we want to build a neural net to predict the price of real estate.
To do this, we use the following dataset which gives the real estate price in the Toronto area: https://www.kaggle.com/datasets/mangaljitsingh/torontoproperties


## Data Cleaning

First of all, we need to clean the data (with data_cleaning.py).
1) We add the longitude and latitude of every property
2) We drop the row where no longitude and latitude are found
3) We drop columns with no helpful data

## First look at the data

To take a first look at the data, we plot it on a map with different colours depending on the price.

<p align='center'>
<img width="724" alt='Toronto property prices distribution neural net' src="https://user-images.githubusercontent.com/96018383/203926761-c1853666-f0c0-4d29-beb8-c9e2d8278219.png">
</p>

Here is the same data when we zoom in on Toronto Downtown :

<p align='center'>
<img width="724" alt='Toronto property prices distribution neural net' src="https://user-images.githubusercontent.com/96018383/203926765-25389eb4-b871-4896-aaee-9e1679d2a996.png">
</p>

We can also look at the price distribution of the property (with a log scale for the y-axis):
<p align='center'>
<img alt='Toronto property prices distribution neural net' src='https://user-images.githubusercontent.com/96018383/203933963-7ebcf916-7d2c-4603-a4c5-bac250367046.png'>
</p>

From this figure, we can see that most of the real estate price is between $0 and $500,000. The frequency of apparition is inverlsely correlated with the price.

We can also look a the price against the number of rooms:
<p align='center'>
  <img alt='Toronto property prices distribution neural net' src='https://user-images.githubusercontent.com/96018383/203933914-72c5ffae-772d-47cd-b64a-b82d0d61a8fe.png'>
</p>

From this figure, we can see that the more rooms we have, the higher the price is. We can also notice that some properties don't have any rooms.

## Creation of new features

So far, we only have 4 features to guess the price of a property: the number of bedrooms, the number of bathrooms, the latitude and the longitude.
We create  4 more features: the number of rooms (bedrooms + bathrooms), the polar coordinates r and $\theta$ (from the latitude and longitude) with a center in downtown Toronto and we map the region to a different number.

## Neural Network

We use a neural network with 5 dense layers with 2048, 1024, 512, 256 and 128 neurons.
After training over 1000 epochs, we get the following results for the loss function.

<p align='center'>
  <img alt='Toronto property neural net' src='https://user-images.githubusercontent.com/96018383/203937654-84d3bdf0-8fe2-45ad-8c74-5d2919ef1499.png'>
 </p>
 

 
 From the above figure, we can see that the loss function stop improving on the testing set which means we can't improve the result by adding more epochs.

To calculte the error, we considere a price match if the predicted price in with +/- x% where x is a variable between 5% and 30% (an acceptable value for x would be 10% but we want to see the results with higher value of x).
We also calculte the error before training our model to get a randome error baseline : 
 
| Margin of error for <br>a correct guess (x) | Correct random guess  | Correct guess with V1 training dataset | Correct guess with V1 test dataset |
|---|---|---|---|
| +/- 5 % | 6 % | 20 % | 20 % |
| +/- 10 % | 15 % | 40 % | 38 % |
| +/- 15 %  | 22 % | 56 % | 52 % |
| +/- 20 % | 28 % | 70 % | 62 % |
| +/- 30 % | 47 % | 87 % | 81 % |

We can see a good price prediction within a +/- 30% margin but the price prediction is not that great within a +/- 10% margin of error.

This could due to 2 reasons : 
First, we only have around 5000 properties(split between training and testing) and second, we have few data points for each property. We only have rooms information and the property location but we don't know if it's a house or an apartment, the number of floors, the sqft, whether it has a garden or not, what is the view, etc.
 
 If we try with a simple regression (with a gradient descent) we have similar results (regression.py)
 
 Conclusion: we need more and better data to improve this model.
 


## Neural net with better data

For the V2, we use the following data set : https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset
This dataset cointains real estate properties all over the US and have 2 additional parameters compare to our previous dataset : the house size and the lot size.

To simplify our problem, we only keep data from New York City and its surrounding.
We also filter out rows with missing data for the bedrooms, bathrooms, house size and lot size.



After cleaning, i.e. removing duplicate rows and rows with missing data and rows where the latitude and longitude were not found, we only have 1200 properties left.

<p align='center'>
  <img width="724" alt='New York property prices neural net' src='https://user-images.githubusercontent.com/96018383/204666557-2cd5ac1a-e2d9-4d25-a186-d945fe11b1fa.png'>
 </p>

We can also see the price distribution : 

<p align='center'>
  <img alt='New York property prices distribution neural net' src='https://user-images.githubusercontent.com/96018383/204667377-b4646145-b6b5-48d1-a8b1-ece2d305e92e.png'>
 </p>

Here is the loss on the training and validation set during training.

<p align='center'>
  <img alt='New York property prices distribution neural net loss' src='https://user-images.githubusercontent.com/96018383/204670844-ee8b2d6a-0cc5-48c1-b08e-cdfadf4606e1.png'>
 </p>

We can see the loss quickly improves on the validation set but then rise up. We overfitted the data.

Here is the error we get at 25 epochs : 

| Margin of error for  a correct guess (x) | Correct random guess  | Correct guess with V2 training dataset (without lat and long) | Correct guess with V2 test dataset (without lat and long) | Correct guess with V2 training dataset (with lat and long) | Correct guess with V2 test dataset (with lat and long) |
|---|---|---|---|---|---|
| +/- 5 % | 4 % | 8 % | 8 % | 7 % | 6 % |
| +/- 10 % | 8 % | 17 % | 18 % | 34 % | 30 % |
| +/- 15 %  | 12 % | 24 % | 27 % | 48 % | 42 % |
| +/- 20 % | 16 % | 31 % | 30 % | 60 % | 54 % |
| +/- 30 % | 25 % | 47 % | 47 % | 77 %  | 73 % |

