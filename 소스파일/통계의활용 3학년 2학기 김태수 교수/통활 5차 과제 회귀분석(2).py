# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 19:17:00 2022

@author: Leejongwoong
"""
import pandas as pd
import math
import numpy as np
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Python_Data_Science_Statical_Study_example_source/Seoultech_University/Statistic_practice/2.수능과 대학성적.csv                                                               ")

# [high_GPA]와 [univ_GPA]의 산점도 그리기
plt.scatter(df['high_GPA'], df['univ_GPA'])
plt.title('high_GPA & univ_GPA')
plt.xlabel('high_GPA')
plt.ylabel('univ_GPA')
plt.show()

# [high_GPA]와 [univ_GPA]의 상관계수 구하기
print(stats.pearsonr(df['high_GPA'],df['univ_GPA']))

# [high_GPA]를 이용하여 [univ_GPA]를 예측하기

result1 = sm.OLS(df['univ_GPA'], sm.add_constant(df['high_GPA'])).fit()
print(result1.summary())

print()
print(np.mean(df['math_SAT']),"\n")
print(np.mean(df['verb_SAT']),"\n")
print(np.std(df['math_SAT']),"\n")
print(np.std(df['verb_SAT']),"\n")
print(stats.pearsonr(df['math_SAT'],df['verb_SAT']),"\n")
print(stats.pearsonr(df['univ_GPA'],df['comp_GPA']),"\n")


print()
print()
print(np.mean(df['univ_GPA']),"\n")
print(np.mean(df['comp_GPA']),"\n")


# 다중회귀 부분 여긴 OLS 가 x하나만 돼서 sckit-learn 이용.
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(df[['math_SAT','verb_SAT']],df['univ_GPA'])
y_pred = reg.predict(df[['math_SAT','verb_SAT']])
print("회귀계수 : ", reg.coef_)
print()
print("절편 : ", reg.intercept_)


y_true = df['univ_GPA'].values
SST = np.sum((y_true - np.mean(df['univ_GPA']))**2)
SSR = np.sum((y_pred - np.mean(df['univ_GPA']))**2)
SSE = SST - SSR
R_score = SSR / SST

# 이제 각 회귀계수에 대한 p-value 구해야하는데 멘붕에 빠짐.

# OLS 다중 되는거 처음 앎.

# 데이터 불러오기

# Univ_GPA = df.drop(['univ_GPA'], axis=1)

# math_SAT,verb_SAT 을 통한 다중 선형회귀분석
x_data = df[["math_SAT","verb_SAT"]] #변수 여러개
target = df[["univ_GPA"]]



# OLS 검정
multi_model = sm.OLS(target, x_data)
fitted_multi_model = multi_model.fit()
print()
print()
print(fitted_multi_model.summary())


# for b0, 상수항 추가할 경우. 

x_data1 = sm.add_constant(x_data, has_constant = "add")

multi_model = sm.OLS(target, x_data1)
fitted_multi_model = multi_model.fit()
print()
print()
print(fitted_multi_model.summary())