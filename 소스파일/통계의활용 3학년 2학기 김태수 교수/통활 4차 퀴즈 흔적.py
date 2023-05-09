# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:45:42 2022

@author: Leejongwoong
"""

import pandas as pd
import math
import numpy as np
import scipy as sp
from scipy import stats
import statsmodels.api as sm
from numpy import linalg # 선대
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({'X':[1,2,3,6,8,10],'Y':[4,6,10,12,16,18]})
reg = LinearRegression()
reg.fit(df[['X']],df['Y'])
y_pred = reg.predict(df[['X']])

X = df['X'].values
Y = df['Y'].values
results_4 = sm.OLS(Y, sm.add_constant(X)).fit()
print(results_4.summary())

SST = np.sum((Y - np.mean(Y))**2)
SSR = np.sum((y_pred - np.mean(Y))**2)
SSE = SST - SSR # SSE = np.sum(((Y - y_pred))**2) 이렇게도 가능
MSR = SSR / 1
MSE = SSE / (len(X)-2)