# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 22:08:14 2020

@author: user02
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM 
from sklearn.preprocessing import MinMaxScaler


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def lstm(gbest_parameter,modeltype,lstm_iter_numr):
    look_back =gbest_parameter[0]
    a=gbest_parameter[1]
    b=gbest_parameter[2]
    batchsize=gbest_parameter[3]
    dataframe = pd.read_csv('..\\data\\typeAOI.csv')
    modeldata = dataframe[['wd','fs','js',modeltype]]  
    dataset = modeldata.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    
    # split into train and test sets
    train_size = int(len(dataset) * 0.8)
    #test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    train = series_to_supervised(train, look_back, 1)
    test = series_to_supervised(test, look_back, 1)
    # reshape into X=t and Y=t+1
    trainXys, trainY = train.values[:, :-4], train.values[:, -1]
    #testXys, testY = test.values[:, :-4], test.values[:, -4]
    
    #
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainXys, (trainXys.shape[0],look_back,4))
    #testX = np.reshape(testXys, (testXys.shape[0], look_back,4))
    # create and fit the LSTM network
    
    model = Sequential()
    #model.add(LSTM(10, input_shape=(look_back,4)))
    model.add(LSTM(a, input_shape=(look_back,4),return_sequences=True))
    model.add(LSTM(b,return_sequences=False))
    
    #model.add(Dropout(0.2)),return_sequences=False))
    #model.add(Dropout(0.2))
    
    model.add(Dense(1))
    #model.add(Dropout(0.2))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history=model.fit(trainX, trainY, epochs=lstm_iter_num, batch_size=batchsize, verbose=1)
    plt.plot(history.history['loss'])
    model.save("..\\result\\model.h5")
    return("model.h5")


modeltype = "zzqo"
#输入的参数为qpso.py运行后返回的参数，每次代码运行前，此处的参数都需要修改     
gbest_parameter = [6, 30, 1, 1]
lstm_iter_num=20
modelname=lstm(gbest_parameter,modeltype,lstm_iter_num)
