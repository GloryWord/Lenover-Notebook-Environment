# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 06:41:16 2022

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

# X:키(inch), Y:몸무게(lb)

df = pd.DataFrame( {'height':[69,56.5,65.3,62.8,63.5,57.3,59.8,62.5,62.5,59,51.3,64.3,56.3,66.5,72,64.8,67,57.5,66.5],
'weight':[112.5,84,98,102.5,102.5,83,84.5,112.5,84,99.5,50.5,90,77,112,150,128,133,85,112]})

reg = LinearRegression()
reg.fit(df[['height']],df['weight'])
y_pred = reg.predict(df[['height']])

X = df['height'].values
Y = df['weight'].values

SST = np.sum((Y - np.mean(Y))**2)
SSR = np.sum((y_pred - np.mean(Y))**2)
SSE = SST - SSR # SSE = np.sum(((Y - y_pred))**2) 이렇게도 가능
MSR = SSR / 1
MSE = SSE / (len(X)-2)
F = MSR / MSE 





# F통계량에 대한 p-value는 직접 구하지 못해서 패키지 이용한다.
results = sm.OLS(Y, sm.add_constant(X)).fit()
print(results.summary())
P_value_F_statistic= 7.89e-7 # summary 표에서 발췌했다.








# 회귀직선의 기울기에 대한 95% 신뢰구간 구하기.
from scipy.stats import t # 이걸로 t 분포표의 값을 끌어내야함.
df = len(X)-2
t_ = t(df) # t가 이미 예약어이기 때문에 t분포표 호출로 t_를 사용

# 회귀분석 분산분석에서는 양측 검정만 가능하다. 그래서 신뢰구간 95% -> 유의수준 5% 따라서 한쪽만 참고하는 0.25로 해야한다.
t_025 = t_.ppf(0.975)

# 맙소사... 회귀직선 기울기의 신뢰구간에서는 연산의 주체가 Y가 아닌 X여야 한다!
lower_limit = reg.coef_ - t_025*(np.sqrt((MSE / np.sum((X-np.mean(X))**2))))
upper_limit = reg.coef_ + t_025*(np.sqrt((MSE / np.sum((X-np.mean(X))**2))))
print()
print("회귀직선의 기울기에 대한 95% 신뢰구간은, {} < {} < {}".format(lower_limit, np.mean(X), upper_limit)) ## 결과: 68.32 < 70 < 70.2









# 회귀직선 그려보기
import matplotlib.pyplot as plt
plt.scatter(X,Y)
regline = reg.coef_*X + reg.intercept_
plt.plot(X, regline, color='red')
plt.title('y = {}*x + {}'.format(reg.coef_, reg.intercept_))
plt.show()