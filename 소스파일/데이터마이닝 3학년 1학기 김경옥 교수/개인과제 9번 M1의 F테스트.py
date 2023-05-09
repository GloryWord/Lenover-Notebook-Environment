# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:38:45 2022

@author: 이종웅
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import t, f, probplot
%matplotlib inline
data = pd.read_csv('https://drive.google.com/uc?export=download&id=1Bs6z1GSoPo2ZPr5jL2qDjRghYcMUOHbS')

M1 = LinearRegression()
M1.fit(data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']], data['quality'])

y_pred_M1 = M1.predict(data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']])

y_true_M1 = data['quality'].values

SST = np.sum((y_true_M1 - np.mean(y_true_M1))**2)
SST
SSR = np.sum((y_pred_M1 - np.mean(y_true_M1))**2)
SSE = SST - SSR

n = len(data)
p = data.shape[1] - 1 # 종속변수 quality가 포함된걸 빼기위해
MSR =  SSR/p
MSE = SSE/(n-p-1)


ftest = MSR/MSE
from scipy import stats # scipy의 stats는 문제에서 주어졌으므로 문제 없을거라고 생각했습니다.
1-stats.f.cdf(ftest,p,n-p-1) #1.1102230246251565e-16


# 모델 2


# 모델 M2을 학습하라 했으므로
M2 = LinearRegression()
M2.fit(data[['volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'pH', 'sulphates', 'alcohol']], data['quality'])

y_pred_M2 = M2.predict(data[['volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide','pH', 'sulphates', 'alcohol']])

y_true_M2 = data['quality'].values

SST2 = np.sum((y_true_M2 - np.mean(y_true_M2))**2)
SSR2 = np.sum((y_pred_M2 - np.mean(y_true_M2))**2)
SSE2 = SST2 - SSR2

n_2 = len(data)
p_2 = data.shape[1] - 1 - 2 #변수 두 개를 뺐음.
MSR2 =  SSR2/p_2
MSE2 = SSE2/(n_2-p_2-1)

ftest2 = MSR2/MSE2
from scipy import stats
p_value_M2 = 1-stats.f.cdf(ftest2,p_2,n_2-p_2-1) #1.1102230246251565e-16 
print(p_value_M2,"\n") 


# 모델 M3

M3 = LinearRegression()
M3.fit(data[['volatile acidity', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'pH', 'sulphates', 'alcohol']], data['quality'])

y_pred_M3 = M3.predict(data[['volatile acidity', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide','pH', 'sulphates', 'alcohol']])

y_true_M3 = data['quality'].values

SST3 = np.sum((y_true_M3 - np.mean(y_true_M3))**2)
SSR3 = np.sum((y_pred_M3 - np.mean(y_true_M3))**2)
SSE3 = SST3 - SSR3

n_3 = len(data)
p_3 = data.shape[1] - 1 - 2 -2 #변수 네 개를 뺐음.
MSR3 =  SSR2/p_3
MSE3 = SSE2/(n_3-p_3-1)

ftest3 = MSR3/MSE3
from scipy import stats
p_value_M3 = 1-stats.f.cdf(ftest3,p_3,n_3-p_3-1) #1.1102230246251565e-16 
print(p_value_M3,"\n") 