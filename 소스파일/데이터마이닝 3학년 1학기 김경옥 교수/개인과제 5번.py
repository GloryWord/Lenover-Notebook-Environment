# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:43:39 2022

@author: 이종웅
"""

# packages and data import
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import t, f, probplot
from scipy import stats
%matplotlib inline
data = pd.read_csv('https://drive.google.com/uc?export=download&id=1Bs6z1GSoPo2ZPr5jL2qDjRghYcMUOHbS')

# (5) 전체 설명변수를 모두 이용해서 선형회귀 모델 M1을 학습하고 그 결과를 바탕으로 아래 표를 채우시오. 

# 모델 M1을 학습하라 했으므로
M1 = LinearRegression()
M1.fit(data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']], data['quality'])
print("절편은",M1.intercept_, "이다 ")
coef = list[M1.coef_]

print("추정 계수들은",coef, "이다")



# 표준오차를 구하는 과정

y_pred_M1 = M1.predict(data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']])
y_true_M1 = data['quality'].values

SSE = np.sum((y_true_M1 - y_pred_M1)**2)
n = len(data)
p = data.shape[1] - 1
MSE = SSE/(n-p-1)


X= data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']].values
X.shape[1]
X = np.c_[np.ones(n), X]
XtX = np.matmul(X.T, X)
from numpy import linalg
inv_XtX = linalg.inv(XtX)


beta = np.matmul(np.matmul(inv_XtX, X.T), data[['quality']].values)

#se(beta)
se_beta = np.sqrt(np.diag(MSE*inv_XtX))
se_beta


alpha = 0.05
stats.t.ppf(1-alpha/2, n-p-1)

# t값
t = []
for i in range(0,12):
    t.append(beta[i][0]/se_beta[i])


# P값
p_value_t =[]
for j in range(0,12):
    p_value_t.append(2*(1-stats.t.cdf(np.abs(t[j]), n-p-1)))



#-------------------------------------치트키

import statsmodels.api as sm

# 데이터 불러오기

Cheat = data.drop(['quality'], axis=1)

# crim, rm, lstat을 통한 다중 선형회귀분석
x_data = data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']] #변수 여러개
target = data[["quality"]]

# for b0, 상수항 추가
x_data1 = sm.add_constant(x_data, has_constant = "add")

# OLS 검정
multi_model = sm.OLS(target, x_data1)
fitted_multi_model = multi_model.fit()
fitted_multi_model.summary()


