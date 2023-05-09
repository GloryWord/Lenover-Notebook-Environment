# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:08:03 2022

@author: 이종웅
"""

import pandas as pd
import matplotlib.pyplot as plt

sales= pd.read_csv('https://drive.google.com/uc?export=download&id=1n9SDdK2pFbM0H14ZSRLB8HpreFYBl6KH')

plt.scatter(sales['temperature'], sales['sales'])

from sklearn.linear_model import LinearRegression


reg = LinearRegression()
reg.fit(sales[['temperature']], sales['sales']) #선형회귀식에서 fit()은 (독립변수, 종속변수), 단 다변수일 경우를 대비해 독립변수는 [[]]로 쓰기.

sales[['temperature']].shape #12행 1열. 이게 DataFrame sales의 크기가 아닌 temperature열의 크기이다.

reg.coef_ #fit 함수를 실행하지 않고 해서 AttributeError: 'LinearRegression' object has no attribute 'coef_' 오류가 떴다.
reg.intercept_ # 절편 -159.4741523408623 나옴.

y_pred = reg.predict(sales[['temperature']])


#------------------------------------------------------------------------------
petrol = pd.read_csv(r'C:\Users\이종웅\.spyder-py3\Downloads\petrol_consumption.txt', names=['tax','income','highway','driver','petrol'], sep='\t')
#r은 붙인 이유는 윈도우즈 파일 시스템에서 디렉토리인 \ 해석을 \\로 해야하기 때문에 발생한  오류. 그래서 r을 붙여줘야 \를 \\로 인식함. r 안붙이려면 \ -> \\으로 수정.

reg.fit(petrol[['tax','income','highway','driver']], petrol['petrol']) #다중선형회귀에서 5개의 독립변수, petrol의 종속변수

reg.intercept_# 2.2737367544323206e-13 으로 변경
list(reg.coef_) # 추정계수가 리스트로 쭉 늘려졌다. 왜냐면 다변수니까. 다변수 개수만큼 추정계수 존재.


y_pred = reg.predict(petrol[['tax','income','highway','driver']]) # 다중선형회귀의 예측값 생성.

import numpy as np
from numpy import linalg 

X = petrol[['tax','income','highway','driver']].values

np.ones((12,)) # 열을 생략했기 때문에 1열로 나열됨.
np.ones((12,3))

X1 = np.c_[np.ones(len(X)),X] 

X1.T.shape #(5,48)
X1.shape  #(48,5)

A = np.array([[1,2],[3,4]])
B = np.array([[1,2],[3,4]])

A * B # 행렬의 곱이 아님. 그럼 뭐라고 묻는다면 모르겠다.
np.matmul(A,B)



## 여기부터
XtX = np.matmul(X1.T, X1) # 현재의 추측으로는 Xt를 그냥 X1으로 쓰고 과정만 X1를 활용했으며, XtX구하기가 목적이었던 듯 하다.

XtX.shape # (5,48) 행렬과 (48,5)의 곱이므로 (5,5) 행렬 나옴.

inv_XtX = linalg.inv(XtX)
inv_XtX.shape

I = np.matmul(XtX, inv_XtX) #대각성분이 전부 1인데 이게 뭐지...? 

beta = np.matmul(np.matmul(inv_XtX, X1.T), petrol[['petrol']].values) 

## 여기까지의 과정을 자세하게 공부할 필요가 있다. 모르는 파트이기때문이다.
beta

beta[0]
reg.intercept_

beta[1]
reg.coef_[0]

from scipy import stats

y_pred = reg.predict(petrol[['tax','income','highway','driver']]) # 독립변수들로 y 추정치 계산
y_true = petrol['petrol'].values # 종속변수로 y의 실제값 계산 petrol이 인수로 들어가는게 충격;; -> 네이밍쪽 보기(30줄)

SST = np.sum((y_true - np.mean(y_true))**2) # y- y바 (표본평균인듯)  / 실제 데이터의 전체분산 ,**은 제곱의 의미 
SST # SSR + SSE의 값과 같게 나온다면 성공!

SSR = np.sum((y_pred - np.mean(y_true))**2) # 잔차의 제곱합이고 잔차란, 실제값의 평균 - 예측값 , 회귀직선으로 설명가능한 오차 
SSR

SSE = np.sum((y_true - y_pred)**2)
SSE

p = 4 # 베타의 개수
n = len(petrol) # 48 나옴

MSR = SSR/p  # MSR는 R을 보고 SSR 관련된거고, M은 Mean of
MSE = SSE/(n-p-1) # n은 행크기이다. 행크기가 곧 데이터의 개수, 열은 변수의 개수.

MSR
MSE

ftest = MSR/MSE # 이 분포가 F분포를 따르기 때문에 F test라고 불린다고 함.
ftest # 22.70...... 나온다.

alpha = 0.05 # 유의수준
stats.f.ppf(1-alpha, p,n-p-1) # 분자의 자유도 4 , 분모의 자유도 42, 유의수준 0.05에 해당하는 F분포표에서의 값이 나옴. 왜 ppf썼는지는 외워야 함. 값 : 2.5888361455239277  그리고 이것이 1에 가까울수록 정규분포와 같은 것. 

1-stats.f.cdf(ftest,p,n-p-1) # F검정에 대한 p값 출력. 3.907167922534427e-10 (= 0.00000000039071679225...) 즉 유의수준 0.05에 한참 못미치므로 통계적으로 유의한 결과는 아니다.



    
    
    
