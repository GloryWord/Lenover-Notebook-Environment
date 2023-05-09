# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 12:08:59 2022

@author: Leejongwoong
"""

import pandas as pd
import math
import numpy as np
import scipy as sp
from scipy import stats
import statsmodels.api as sm


X = np.array([50,70,40,90,60,50])
Y = np.array([70,70,60,80,70,50])

A = np.array([20,24,26,30,34,40])
B = np.array([32,30,24,22,20,16])
result = stats.pearsonr(X, Y)
meanX = np.mean(X)
meanY = np.mean(Y)

# 공분산 구하는 함수
def covariance(X, Y): 
    ax, ay = X.mean(), Y.mean()
    data = [round((ax-x)*(ay-y),2) for x, y in zip(X, Y)] # product of deriviations
    print('data:',data)
    return sum(data) / len(X)

covar_XY = covariance(X,Y)
# 모상관계수
origin_correlation = covar_XY / math.sqrt(np.var(X)*np.var(Y))

# 표본상관계수 = 피어슨 상관계수
# 분자와 분모를 따로 구해본다.
upper = sum([((x-meanX)*(y-meanY)) for x,y in zip(X,Y)])


# 분모 구하는 과정
X_deviation_square_sum = 0
for x in X:
    X_deviation_square_sum = X_deviation_square_sum + ((x-meanX)**2)

Y_deviation_square_sum = 0
for y in Y:
    Y_deviation_square_sum = Y_deviation_square_sum + ((y-meanY)**2)
 
down = math.sqrt(X_deviation_square_sum * Y_deviation_square_sum)

# 표본상관계수 구하기
r_XY = upper / down

# scipy로 구하기

print(stats.pearsonr(X, Y))

# 결론 : 모상관계수와 표본상관계수(피어슨 상관계수)는 직접계산으로나, 사이파이나, 표본상관계수 공식으로나 같게 나온다.

# 사실 분자(upper)은 n으로 나누면 공분산이었고, n-1로 나누면 표본 공분산이 되는 것이다.

sampled_covar_XY = upper / (len(X)-1)

# 잔머리로 표본분산 구하기. (np.var(X) * n) / (len(X)-1)
sampled_var_X = (np.var(X) * len(X)) / (len(X)-1)
sampled_var_Y = (np.var(Y) * len(Y)) / (len(Y)-1)


