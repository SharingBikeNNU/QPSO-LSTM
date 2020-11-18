# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 22:36:20 2020

@author: user02
"""

from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



#以典型AOI为例应用模型。预测时间选择3月14日。
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

def plot(gbest_parameter,Predict):
    X = []
    Y = []
    for i in range(gbest_parameter[0],24):
        X.append(i + 1)
        Y.append(Predict[i-gbest_parameter[0]])
    plt.plot(X,Y)
    plt.xlabel('Hour',size = 15)
    plt.ylabel('Number of Bike',size = 15)
    plt.title('QPSO-LSTM model predicts number of bike')
    plt.show()   
    
    

def predict(modelname,modeltype,gbest_parameter):
    model=load_model("..\\result\\"+str(modelname))
    look_back=gbest_parameter[0]
    dataframe = pd.read_csv('..\\data\\sampleAOI.csv')
    modeldata = dataframe[['wd','fs','js',modeltype]]  
    dataset = modeldata.values
    dataset = dataset.astype('float32')
            # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(dataset)
    train = series_to_supervised(train, look_back, 1)
    trainX= train.values[:, :-4]
    trainXinput = np.reshape(trainX, (trainX.shape[0],look_back,4))
    Predict = model.predict(trainXinput)
    Predict = np.concatenate((trainX[:, 0:3],Predict), axis=1)
    Predict = scaler.inverse_transform(Predict)[:,3].astype(int)
    plot(gbest_parameter,Predict)
    return Predict

      
#模型预测住宅区（zzo）的o    
modeltype = "zzqo" 
#输入的参数为qpso.py运行后返回的参数，每次代码运行前，此处的参数都需要修改  
gbest_parameter = [6, 30, 1, 1]
modelname="model.h5"
predictnum = predict(modelname,modeltype,gbest_parameter)
#print(predictnum)
