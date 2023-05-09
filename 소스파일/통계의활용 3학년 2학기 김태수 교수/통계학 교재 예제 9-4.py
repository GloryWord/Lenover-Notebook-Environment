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

# 나이와 최대 심장 박동
df = pd.DataFrame( {'Age':[18,23,25,35,65,54,34,56,72,19,23,42,18,39,37],
'Max':[202,186,187,180,156,169,174,172,153,199,193,174,198,183,178]})

reg = LinearRegression()
reg.fit(df[['Age']],df['Max'])
y_pred = reg.predict(df[['Age']])

X = df['Age'].values
Y = df['Max'].values

SST = np.sum((Y - np.mean(Y))**2)
SSR = np.sum((y_pred - np.mean(Y))**2)
SSE = SST - SSR # SSE = np.sum(((Y - y_pred))**2) 이렇게도 가능
MSR = SSR / 1
MSE = SSE / (len(X)-2)
F = MSR / MSE 
Error_list = (Y - y_pred)


# F통계량에 대한 p-value는 직접 구하지 못해서 패키지 이용한다.
results = sm.OLS(Y, sm.add_constant(X)).fit()
print(results.summary())
# P_value_F_statistic= 7.89e-7 # summary 표에서 발췌했다.




# 회귀직선의 기울기에 대한 95% 신뢰구간 구하기.
from scipy.stats import t # 이걸로 t 분포표의 값을 끌어내야함.
degree_free = len(X)-2
t_ = t(degree_free) # t가 이미 예약어이기 때문에 t분포표 호출로 t_를 사용

# 회귀분석 분산분석에서는 양측 검정만 가능하다. 그래서 신뢰구간 95% -> 유의수준 5%, 따라서 한쪽만 참고하는 0.25로 해야한다.
t_025 = t_.ppf(0.975) # ppf는 누적분포 함수의 역함수이다.

# 맙소사... 회귀직선 기울기의 신뢰구간에서는 연산의 주체가 Y가 아닌 X여야 한다!
lower_limit = reg.coef_ - t_025*(np.sqrt((MSE / np.sum((X-np.mean(X))**2))))
upper_limit = reg.coef_ + t_025*(np.sqrt((MSE / np.sum((X-np.mean(X))**2))))
print()
print("회귀직선의 기울기에 대한 95% 신뢰구간은, {} < {} < {}".format(lower_limit, np.mean(X), upper_limit)) ## 결과: 68.32 < 70 < 70.2




# 잔차분석 파트
# 종속변수 y의 평균값의 대한 90% 신뢰구간 -> 유의수준 10%
t_005 = t_.ppf(0.95)
sum_deviation = np.sum((X - np.mean(X))**2)

y_mean_lower_limit = y_pred - (t_005 * np.sqrt(MSE) * np.sqrt(1/len(X) + (((X-np.mean(X))**2)/sum_deviation))) # 와.... 곱하기 sigma hat은 루트 MSE인데, 그냥 MSE를 넣고 왜 안되나 고민했다.

y_mean_upper_limit = y_pred + (t_005 * np.sqrt(MSE) * np.sqrt(1/len(X) + (((X-np.mean(X))**2)/sum_deviation)))

y_pred_lower_limit = y_pred - (t_005 * np.sqrt(MSE) * np.sqrt(1+ (1/len(X)) + (((X-np.mean(X))**2)/sum_deviation)))

y_pred_upper_limit = y_pred + (t_005 * np.sqrt(MSE) * np.sqrt(1+ (1/len(X)) + (((X-np.mean(X))**2)/sum_deviation)))


# 한눈에 보기 위해 표 만들려고 했는데 넘파이 배열들을 데이터 프레임으로 한 곳에 묶는 법을 모르겠다.
#result_error_analysis_chart = pd.DataFrame(y_mean_lower_limit, y_mean_upper_limit, y_pred_lower_limit, y_pred_upper_limit)

