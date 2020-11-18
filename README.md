# QPSO-LSTM
Prediction for OD distribution of dockless shared bicycles: a case study in Nanjing city

## Information of QPAO-LSTM model
The QPSO-LSTM model mainly includes two core algorithms, namely QPSO(quantum particle swarm optimization)(Sun et al. 2004) and LSTM(Long Short-Term Memory)(S. Hochreiter et al. 1997).
(1) QPSO is an improvement on traditional particle swarm optimization (PSO). It is inspired by the theory of quantum physics, that is, all particles have quantum behavior. Based on PSO, this algorithm eliminates the attribute of particle movement direction, that is, the update of particle position is independent of the previous particle position. It is more likely to obtain the global optimal solution or approximate optimal solution and determine the global optimal position(gbest) and part optimal position(pbest) of the particle by increasing the randomness of the particle's position. QPSO has been widely used in many fields such as function optimization. QPSO-LSTM model has some hyperparameters to be optimized, such as time step, number of hidden layer nodes, batch size, etc., as shown in the figure below.
<p align="center"><img src="pic/Figure 1.jpg" width="65%"></p>

QPSO can be used to quickly determine the hyperparameter combination suitable for the time prediction model, so as to effectively improve the accuracy of the prediction model. The flowchart of the optimization of the LSTM model with the QPSO optimization algorithm is shown in the figure below.

<p align="center"><img src="pic/Figure 2.jpg" width="65%"></p>

(2)LSTM is designed based on Recurrent Neural Network(RNN) to solve the problem of gradient disappearance in RNN network training, so as to better learn the dependency of long time series data. Compared with RNN, LSTM has not been used to screen the information of previous moments. As a kind of deep learning algorithm for the prediction of serial data, LSTM has a good effect on the prediction of time series data and is widely used in the prediction of stock price, traffic flow and so on. The structure diagram of LSTM is shown in the figure below.
<p align="center"><img src="pic/Figure 3.jpg" width="65%"></p>

In the QPSO-LSTM model, LSTM is used to predict the origin and destination quantity of dockless shared bicycles. The LSTM construction flow chart is shown in the figure below.

<p align="center"><img src="pic/Figure 4.jpg" width="65%"></p>


## How to run code 
You have to install packages such as pandas, sklearn, keras, numpy, math and matplotlib before running the code. The code of QPSO-LSTM model includes 3 pyhon files, the file named qpso should be the first file to be run, follow the file named train, and the last one named predict. 
(1) About qpso.py
In this file, the parameters needed to be set are shown in function __main__(), including particle_num, iter_num, max_value, min_value and lstm_iter_num. The parameter named modeltype, which means the type of AOI, also needs to be set. Its default value is zzqo, which means the parameters will be optimized for the model applied to predict the bicycle number of origin points in residential areas. Finally, the file will output a list of parameters that have been optimized.

(2) About train.py
In this file, the parameters such as modeltype, gbest_parameter and lstm_iter_num needs to be set. Among them, gbest_parameter is the output of qpso.py.

(3) About predict.py 
In this file, the parameters such as modeltype and gbest_parameter also should be set. In addition, the output file from train.py named model.h5 also needs to be set.