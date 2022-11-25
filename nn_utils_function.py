
import matplotlib.pyplot as plt
import numpy as np



def plot_loss_tf(history):
    losses = history.history['loss']
    plt.plot(losses)
    try:
        val_loss = history.history['val_loss']
        plt.plot(val_loss)
        plt.legend(['loss','test_loss'])
    except:
        pass

    plt.xlabel('epoch')
    plt.ylabel('losses')

def normalize_data(data):
    mean = data.mean()
    max_minus_min = data.max()-data.min()
    data = (data-mean)/max_minus_min
    return data, mean, max_minus_min

def unnormalized_data(norm_price,mean_price,max_minus_min_price):

    return norm_price*max_minus_min_price+mean_price

def calculate_errors(model,X,y,E,price_error,mean_price,max_minus_min_price):
    failed_X = []
    failed_y = []

    test_size = X.shape[0]
    prediction = np.exp(unnormalized_data(model.predict(X),mean_price,max_minus_min_price))
    y = np.exp(unnormalized_data(y,mean_price,max_minus_min_price))



    # for i in range(len(prediction)):
    #     print(prediction[i],y[i])
    #     print('error = ',abs(prediction[i]-y[i])/prediction[i]*100,'%')
    failed_X = [X[i,:] for i in range(len(prediction)) if not prediction[i]-price_error<=y[i]<=prediction[i]+price_error]
    failed_prediction = [prediction[i] for i in range(len(prediction)) if not prediction[i]-price_error<=y[i]<=prediction[i]+price_error]
    failed_y = [y[i] for i in range(len(prediction)) if not prediction[i]-price_error<=y[i]<=prediction[i]+price_error]
    percent_failed = round(len(failed_X)/test_size*100*100)/100
    print(f'{percent_failed} % of failure (or {len(failed_X)} out of {test_size})')

    #append to E_cv or E_train
    E.append(percent_failed)

    return np.array(failed_X),np.array([failed_prediction]).T,np.array([failed_y]).T,E


def carth_to_polar(x,y):

    x0 = 43.647144
    y0 = -79.381204 #Toronto Union station

    r = np.sqrt((x-x0)**2+(y-y0)**2)
    theta = np.arctan((x-x0)/(y-y0))
    return r,theta
