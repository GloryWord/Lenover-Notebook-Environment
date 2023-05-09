# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:12:28 2022

@author: 이종웅
"""

import pandas as pd
import numpy as np
from numpy import linalg
from scipy import stats

petrol=pd.read_csv('https://drive.google.com/uc?export=download&id=1R9B0D_fSjfCiSaS1WWEjEbSHFOXlVjQA')

n = len(petrol) # n = 48


# 회귀계수인 beta_hat 만들기. beta_hat = inverse(Xt * X) * Xt * y 외워야 하는 공식임.


X = petrol[['tax', 'income', 'highway', 'license']].values # 해당 열의 모든 행 값들이므로 이중괄호 안에 있음. 

p = X.shape[1] # p = 5. 아마도 미지수 개수를 의미한다. (열개수)
X.shape # (48,5)
X = np.c_[np.ones(n), X] # 두 개의 1차원 배열을 칼럼으로 세로로 붙여서 2차원 배열 만들기

XtX = np.matmul(X.T, X)
XtX.shape # (5,5) 크기의 행

inv_XtX = linalg.inv(XtX)

beta = np.matmul(np.matmul(inv_XtX, X.T), petrol[['consumption']].values)
beta.shape # (5,1)
beta # 회귀계수의 추정치는 기울기의 추정치. 기울기라는건 미지수 갯수와 똑같겠구나. y= beta0 + beta1 x + beta2 x + beta3 x + ....이런식.
beta[1][0]

y_pred = np.matmul(X, beta)
y_pred.shape # (48,1)
y_pred = y_pred.flatten()
# flatten()는, X = np.array([[51, 55], [14, 19], [0, 4]])를 [51 55 14 19 0 4] 으로 1차원으로 평탄
y_pred.shape # (48,) 

SSE = np.sum((petrol['consumption'].values - y_pred) ** 2) # 시그마(실제값 - 추정치)^2 이것이 Sum of Square Error.
MSE = SSE/(n-p-1) # MSE구하는 공식 암기.

cov_beta = MSE * inv_XtX
cov_beta

M1 = LinearRegression()
M1.fit(petrol[['tax', 'income', 'highway', 'license']], petrol['consumption'])
coef = list[M1.coef_]
coef

































A = np.array([[1,2], [3,4]])
A # 2행 2열짜리 행렬
np.diag(A) # 대각성분을 다 뽑아서 1차원 리스트로.
np.diag([1,2,3]) # () 안에 있는 것을 가지는 대각행렬 생성. 위와의 차이점은 위는 인수가 2차원행렬이었다.

np.diag(cov_beta) # 5x5 행렬이 1차원 5개의 요소를 갖는 리스트로 변
# se_beta = np.sqrt(np.diag(cov_beta)) 
se_beta # 이제 t값을 구하는 공식은 외워야 하는 영역이다.

t1 = beta[1][0]/se_beta[1] # 존나 심오하다. 0행부터 시작한다고 가정해야되고, 1행 0열 즉 1행의 첫번째 원소 추출. 0행은 절편이므로 계수(기울기)검정에서 빠진듯 하다.

t1 

alpha = 0.05
stats.t.ppf(1-alpha/2,n-p-1) # 42 자유도를 갖고 0.025의 유의수준을 갖는 t분포표의 값 . 2.0166921941428133  0에 가까울수록 정규분포와 유사하게 됨.
2*(1-stats.t.cdf(np.abs(t1), n-p-1))





































# 일단 과제를 위해 R-squre와 VIF 를 구해보자.

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

reg = LinearRegression()
reg.fit(petrol[['tax', 'income', 'highway', 'license']], petrol['consumption'])
r2 = reg.score(petrol[['tax', 'income', 'highway', 'license']], petrol['consumption']) # score함수가 R^2 구해주는 거임.
#R^2는 독립변수가 종속변수를 얼마나 설명해주느냐.

r2 # 0.6786867115698209 , consumption에 대한 R^2 값 적합성은 약 67%

consumption_vif = 1/(1-r2)
consumption_vif # 3.1122273370193914 가 나오는데, 종속변수의 vif는 필요하지 않다.


cols = ['license', 'tax', 'income', 'highway']

cols[:1] # ['license']
cols[:2] # ['license', 'tax']
cols[:3] # ['license', 'tax', 'income']
cols[:4] # ['license', 'tax', 'income', 'highway']
r2 =[]


# F9 실행은 Shift + Enter랑 컴파일러 상에서 다르므로 내 컴퓨터에서는 for, if는 후자로 한다.
for i in range(len(cols)): # range의 시작점이 없으므로 0부터 len(cols)인 4까지 한다. 총 0,1,2,3 (4번 시행) 
    reg.fit(petrol[cols[:i+1]], petrol['consumption'])
    r2.append(reg.score(petrol[cols[:i+1]], petrol['consumption'])) # 위에는 독립변수가 4개이므로 r2값 역시 변수별로 4개 있어야 함. 그런데 위의 r2는 하나밖에 나오지 않음.

r2 #  [0.4885526602607474, 0.556681093224823, 0.6748583388195666, 0.6786867115698209]


#-----------------------------------------------------------------------------------------

reg.fit(petrol[['income', 'highway', 'license']], petrol['tax'])
r2 = reg.score(petrol[['income', 'highway', 'license']], petrol['tax'])
r2
tax_vif = 1/(1-r2)
tax_vif # 1.6256755482376715

reg.fit(petrol[['tax', 'highway', 'license']], petrol['income'])
r2 = reg.score(petrol[['tax', 'highway', 'license']], petrol['income'])
r2
income_vif = 1/(1-r2)
income_vif # 1.0432741252316342

reg.fit(petrol[['tax', 'income', 'license']], petrol['highway'])
r2 = reg.score(petrol[['tax', 'income', 'license']], petrol['highway'])
r2
highway_vif = 1/(1-r2)
highway_vif # 1.4969365918376891

reg.fit(petrol[['tax', 'highway', 'income']], petrol['license'])
r2 = reg.score(petrol[['tax', 'highway', 'income']], petrol['license'])
r2
license_vif = 1/(1-r2)
license_vif # 1.2163550777459626  

# ------------이곳은 모듈로 VIF 구하기--------------(답지 용)
from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF = pd.DataFrame()

lst = [] # X의 vif값이 들어있는 리스트
for i in range(X.shape[1]):
    v = variance_inflation_factor(X, i) # 컬럼을 정수로 지정한다.
    lst.append(v)

VIF["VIF values"] = lst
VIF["features"] = petrol.columns
VIF
#     VIF values     features
# 0  375.849315          tax
# 1    1.625676       income
# 2    1.043274      highway
# 3    1.496937      license
# 4    1.216355  consumption    나온다.
















import statsmodels.api as sm

# 데이터 불러오기

Jwdata = petrol.drop(['consumption'], axis=1)

# crim, rm, lstat을 통한 다중 선형회귀분석
x_data = petrol[['tax', 'income', 'highway', 'license']] #변수 여러개
target = petrol[["consumption"]]

# for b0, 상수항 추가
x_data1 = sm.add_constant(x_data, has_constant = "add")

# OLS 검정
multi_model = sm.OLS(target, x_data1)
fitted_multi_model = multi_model.fit()
fitted_multi_model.summary()






