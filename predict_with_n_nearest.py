#Use the n nearest properties to predict the real estate price per sqft


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nn_utils_function import *
from sklearn.model_selection import train_test_split



df = pd.read_csv('cleaned_realtor_data_w_lat_long.csv')
df = df.fillna('')
df = df[df['zip_code']!='']


df['price_per_sqft'] = df['price']/df['house_size']

# We split the data between a training set and a test set but we don't use a neural net
x_df = df[['bed','bath','acre_lot','house_size','latitude','longitude','zip_code','r','theta','price_per_sqft']]
X = x_df.to_numpy().astype('float32')
y = df['price_per_sqft'].to_numpy().astype('float32')

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15, random_state=1)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)




# For a given property p in the test set
# Step 1 : find the n nearest properties
# Step 2 : calculate the average price in the area


def find_n_nearest(X_train,p,n)->list:

    x_0 = p[4] #latitude
    y_0 = p[5] #longitude
    from heapq import heappop, heappush, heapify,nlargest
    heap = []
    heapify(heap)

    for i in range(X_train.shape[0]):
        x = X_train[i,4]
        y = X_train[i,5]
        r = np.sqrt((x-x_0)**2+(y-y_0)**2)
        heappush(heap,(-1*r, X_train[i,:].tolist()))
        if len(heap)==n+1:
            heappop(heap)

    return [s[1] for s in nlargest(n,heap)]

def find_average_price(n_nearest,n):
    avg = 0
    #print(n_nearest)
    for i in range(n):
        #print(n_nearest[i][9])
        avg += n_nearest[i][9]
    return avg / n

## calcul the error
n_vec = range(1,30,2)
p_error = [0.01,0.05,0.10,0.15,0.2,0.3]
error_matrix = []
len_test = X_test.shape[0]
for n in range(1,30,2):
    print(n)

    #1) Calcul the prediction price
    predicted_test_price = []
    for i in range(len_test):

        predicted_test_price.append(find_average_price(find_n_nearest(X_train,X_test[i,:],n),n))
        if i%(len_test//10)==0:
            pct = i/(len_test)*100
            print(f'{int(pct)}%')
    #2) Calculate the error
    p_vec = []
    for p in p_error:
        correct = 0

        for i in range(len_test):
            real_price_per_sqft = X_test[i,9]
            predicted_test_price_per_sqft = predicted_test_price[i]
            if predicted_test_price_per_sqft-p*predicted_test_price_per_sqft<=real_price_per_sqft<=predicted_test_price_per_sqft+p*predicted_test_price_per_sqft:
                correct += 1
        correct_pct = correct/len_test*100
        p_vec.append(correct_pct)
        print(f'We have a {correct_pct:0.2f}% correct match within an error margin of {p*100}% for n = {n}')
    error_matrix.append(p_vec)

error_matrix = np.array(error_matrix)
legend = []
for i in range(len(p_error)):
    err = error_matrix[:,i]
    print(n_vec)
    print(err)
    plt.plot(n_vec,err)
    legend.append(str(p_error[i]*100)+'%')

plt.xlabel('Number of neighbours n')
plt.ylabel('Prediction accuracy (%)')
plt.legend(legend)
plt.title('Prediction accuracy for different margins of error')
plt.show()
