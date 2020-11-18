# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 20:25:42 2020

@author: user02
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
#from pandas import read_csv
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

## 1.加载数据

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
def MAPE(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return (1-np.mean(diff / true))

## 2. QPSO算法
class QPSO(object):
    def __init__(self,particle_num,particle_dim,alpha,iter_num,max_value,min_value,modeltype,lstm_iter_num):
        '''定义类参数
        particle_num(int):粒子群大小
        particle_dim(int):粒子维度，对应待寻优参数的个数
        alpha(float):控制系数
        iter_num(int):最大迭代次数
        max_value(float):参数的最大值
        min_value(float):参数的最小值
        '''
        self.particle_num = particle_num
        self.particle_dim = particle_dim
        self.iter_num = iter_num
        self.alpha = alpha
        self.max_value = max_value
        self.min_value = min_value

### 2.1 粒子群初始化
    def swarm_origin(self):
        '''初始化粒子群中的粒子位置
        input:self(object):QPSO类
        output:particle_loc(list):粒子群位置列表
        '''
        particle_loc = []
        for i in range(self.particle_num):
            tmp1 = []
            for j in range(self.particle_dim):
                a = random.random()
                tmp1.append(a * (self.max_value[j] - self.min_value[j]) + self.min_value[j])
            particle_loc.append(tmp1)
        return particle_loc

### 2.2 计算适应度函数数值列表
        
    
    def fitness(self,particle_loc,modeltype,lstm_iter_num):
        fitness_value = []
        ### 1.适应度函数为RBF_SVM的3_fold交叉校验平均值
        for i in range(self.particle_num):
            look_back =int(particle_loc[i][0])
            a=int(particle_loc[i][1])
            b=int(particle_loc[i][2])
            batchsize=int(particle_loc[i][3])
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
            testXys, testY = test.values[:, :-4], test.values[:, -1]
            
            #
            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainXys, (trainXys.shape[0],look_back,4))
            testX = np.reshape(testXys, (testXys.shape[0], look_back,4))
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
            model.fit(trainX, trainY, epochs=lstm_iter_num, batch_size=batchsize, verbose=1)
            # make predictions
            #trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)
            
            testPredict = np.concatenate((testXys[:, 0:3],testPredict), axis=1)
            testY = np.concatenate((testXys[:, 0:3],testY.reshape(-1,1)), axis=1)
            
            # invert predictions
            #trainPredictSC = scaler.inverse_transform(trainPredict)
            #trainYSC = scaler.inverse_transform([trainY])
            testPredictSC = scaler.inverse_transform(testPredict)
            testYSC = scaler.inverse_transform(testY)
            # calculate root mean squared error
            # =============================================================================
           #trainScore = math.sqrt(mean_squared_error(testYSC[:,0], testPredictSC[:,0]))
            trainScore=MAPE(testYSC[:,3], testPredictSC[:,3])
            fitness_value.append(trainScore)
            current_fitness = 0.0
            current_parameter = []
        for i in range(self.particle_num):
            if current_fitness < fitness_value[i]:
                current_fitness = fitness_value[i]
                current_parameter = particle_loc[i]

        return fitness_value,current_fitness,current_parameter
            # print('Train Score: %.2f RMSE' % (trainScore))
            # =============================================================================       
     ### 2.3 粒子位置更新    
    def updata(self,particle_loc,gbest_parameter,pbest_parameters):
        '''粒子位置更新
        input:self(object):QPSO类
              particle_loc(list):粒子群位置列表
              gbest_parameter(list):全局最优参数
              pbest_parameters(list):每个粒子的历史最优值
        output:particle_loc(list):新的粒子群位置列表
        '''
        Pbest_list = pbest_parameters
        #### 2.3.1 计算mbest
        mbest = []
        total = []
        for l in range(self.particle_dim):
            total.append(0.0)
        total = np.array(total)
        
        for i in range(self.particle_num):
            total += np.array(Pbest_list[i])
        for j in range(self.particle_dim):
           mbest.append(list(total)[j] / self.particle_num)
        
        #### 2.3.2 位置更新
        ##### Pbest_list更新
        for i in range(self.particle_num):
            a = random.uniform(0,1)
            Pbest_list[i] = list(np.array([x * a for x in Pbest_list[i]]) + np.array([y * (1 - a) for y in gbest_parameter]))
        ##### particle_loc更新
        for j in range(self.particle_num):
            mbest_x = []  ## 存储mbest与粒子位置差的绝对值
            for m in range(self.particle_dim):
                mbest_x.append(int(abs(mbest[m] - particle_loc[j][m])))
            u = random.uniform(0,1)
            if random.random() > 0.5:
                particle_loc[j] = list(np.array(Pbest_list[j]) + np.array([self.alpha * math.log(1 / u) * x for x in mbest_x]))
            else:
                particle_loc[j] = list(np.array(Pbest_list[j]) - np.array([self.alpha * math.log(1 / u) * x for x in mbest_x]))
                
        #### 2.3.3 将更新后的量子位置参数固定在[min_value,max_value]内 
        ### 每个参数的取值列表
        parameter_list = []
        for i in range(self.particle_dim):
            tmp1 = []
            for j in range(self.particle_num):
                tmp1.append(particle_loc[j][i])
            parameter_list.append(tmp1)
        ### 每个参数取值的最大值、最小值、平均值   
        value = []
        for i in range(self.particle_dim):
            tmp2 = []
            tmp2.append(max(parameter_list[i]))
            tmp2.append(min(parameter_list[i]))
            value.append(tmp2)
        
        for i in range(self.particle_num):
            for j in range(self.particle_dim):
                particle_loc[i][j] = (particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (self.max_value[j] - self.min_value[j]) + self.min_value[j]               
        return particle_loc

## 2.4 画出适应度函数值变化图
    def plot(self,results):
        '''画图
        '''
        X = []
        Y = []
        for i in range(self.iter_num):
            X.append(i + 1)
            Y.append(results[i])
        plt.plot(X,Y)
        plt.xlabel('Number of iteration',size = 15)
        plt.ylabel('Value of MAPE',size = 15)
        plt.title('QPSO-LSTM parameter optimization')
        plt.show()         

## 2.5 主函数
    def main(self):
        results = []
        best_fitness = 0.0 
        ## 1、粒子群初始化
        particle_loc = self.swarm_origin()
        ## 2、初始化gbest_parameter、pbest_parameters、fitness_value列表
        ### 2.1 gbest_parameter
        gbest_parameter = []
        for i in range(self.particle_dim):
            gbest_parameter.append(0.0)
        ### 2.2 pbest_parameters
        pbest_parameters = []
        for i in range(self.particle_num):
            tmp1 = []
            for j in range(self.particle_dim):
                tmp1.append(0.0)
            pbest_parameters.append(tmp1)
        ### 2.3 fitness_value
        fitness_value = []
        for i in range(self.particle_num):
            fitness_value.append(0.0)
        
        ## 3、迭代
        for i in range(self.iter_num):
            ### 3.1 计算当前适应度函数值列表
            current_fitness_value,current_best_fitness,current_best_parameter = self.fitness(particle_loc,modeltype,lstm_iter_num)
            ### 3.2 求当前的gbest_parameter、pbest_parameters和best_fitness
            for j in range(self.particle_num):
                if current_fitness_value[j] > fitness_value[j]:
                    pbest_parameters[j] = particle_loc[j]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                gbest_parameter = current_best_parameter
            
            print('iteration is :',i+1,';Best parameters:',list(map(int,gbest_parameter)),';Best fitness',best_fitness)
            results.append(best_fitness)
            print(best_fitness)
            ### 3.3 更新fitness_value
            fitness_value = current_fitness_value
            ### 3.4 更新粒子群
            particle_loc = self.updata(particle_loc,gbest_parameter,pbest_parameters)
        ## 4.结果展示
        results.sort()
        self.plot(results)
        print('Final parameters are :',list(map(int,gbest_parameter)))
        return list(map(int,gbest_parameter))
    
    
if __name__ == '__main__':
    particle_num = 3
    particle_dim = 4
    iter_num = 2
    alpha = 0.6
    lstm_iter_num=20
    max_value = [6,30,30,20]
    min_value = [2,1,1,1]
    #选择构建的模型类型，范围为，["zzqo","zzqd","swlyo","swlyd","kjwho","kjwhd","qtfwo","qtfwd"]
    modeltype = "zzqo"
    qpso = QPSO(particle_num,particle_dim,alpha,iter_num,max_value,min_value,modeltype,lstm_iter_num)
    gbest_parameter=qpso.main()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


