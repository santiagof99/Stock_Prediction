#Will use Reccurent Neural Network Strategy called Long Short Term Memory (LSTM)
#Will take in a Corporation Stock name as a paramter and make a 7 day prediction using previous 2 months as data 

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import math
from keras.models import Sequential 
from keras.layers import LSTM, Dense 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 

plt.style.use('fivethirtyeight')






def get_model_prediction(stock):
    d_frame = pdr.DataReader(stock, data_source='yahoo',start='2013-01-01',end='2020-8-27')
    print(d_frame)
    plt.figure(figsize=(16,8))
    plt.title('Close Price History')
    plt.plot(d_frame['Close'])
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price USD ($)',fontsize=18)
    plt.show()

    #Get the Closing Stock price column only
    d_frame_close = d_frame.filter(['Close'])
    #Make a numpy array from the closing values
    data_set = d_frame_close.values
    #make sure to store number of rows for the model training 
    data_length = math.ceil(len(data_set)*.8)


    #values now between 0 and 1
    MM_Scaler = MinMaxScaler(feature_range=(0,1))
    data_set_scaled = MM_Scaler.fit_transform(data_set)


    #training data set and get the x and y values for the model and make them arrays
    training_data = data_set_scaled[0:data_length,:]
    x_train = []
    y_train=[]

    for x in range(120,len(training_data)):
        x_train.append(training_data[x-120:x,0])
        y_train.append(training_data[x,0])

    x_train,y_train = np.array(x_train),np.array(y_train)


    #since the model needs 3 dimensional data we need to make some converions to include #samples , time steps, and #features
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


    #Build the LSTM 
    L_Model = Sequential()
    L_Model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    L_Model.add(LSTM(units=50,return_sequences=False))
    L_Model.add(Dense(units=25))
    L_Model.add(Dense(units=1))
    #compile model with the adam optimizer to help with the noisy data and the loss of mean squared error
    L_Model.compile(optimizer='adam',loss='mean_squared_error')

    #Model Training
    L_Model.fit(x_train,y_train,batch_size=1,epochs=1)

    #create testing scaled array with x and y data sets
    testing_data = data_set_scaled[data_length-120: , :]
    x_test_set = []
    y_test_set = data_set[data_length:, :]

    for x in range(120,len(testing_data)):
        x_test_set.append(testing_data[x-120:x, 0])

    #data to np array and reshape it to 3D
    x_test_set = np.array(x_test_set)
    x_test_set = np.reshape(x_test_set,(x_test_set.shape[0],x_test_set.shape[1],1))

    #receive predicted price
    model_predictions = L_Model.predict(x_test_set)
    model_predictions = MM_Scaler.inverse_transform(model_predictions)

    rmse=np.sqrt(np.mean(((model_predictions- y_test_set)**2)))
    print(rmse)


    #Plot/Create the data for the graph
    train = d_frame_close[:data_length]
    valid = d_frame_close[data_length:]
    valid['Predictions'] = model_predictions
    #Visualize the data
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()
