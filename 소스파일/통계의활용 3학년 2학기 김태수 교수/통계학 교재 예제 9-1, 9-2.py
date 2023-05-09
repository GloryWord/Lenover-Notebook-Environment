# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 22:31:22 2022

@author: Leejongwoong
"""

import pandas as pd
import math
import numpy as np
import scipy as sp
from scipy import stats
import statsmodels.api as sm

# 예제 9.1 (206p)
# 다음 자료를 근거로 선형 상관계수를 구하고 유의수준 0.01 일때 수출액과 수입액의 두 변수간 관계를 분석하고 그 관계를 수식화하여라. (단위: 10억 달러)

df1 = pd.DataFrame({'years':[1996,1997,1998,1999,2000,2001],'export_revenue':[16,20,27,39,56,63], 'import_revenue':[2,3,4,7,11,11]})

# 선형성을 확인 해보기 위해 상관계수를 구한다.
corr1 = stats.pearsonr(df1['export_revenue'], df1['import_revenue']) 
# (상관계수, p-value) 순이다.
# 맙소사.... sckit 런에서 fit 함수에는 독립변수 2차원 리스트 넣어야 하는걸 
# 통계 패키지일 뿐인 stats에다가 2차원을 적용해서 ValueError: shapes (6,1) and (6,) not aligned: 1 (dim 1) != 6 (dim 0)가 발생했다. 여긴 1차원 리스트만 넣어라.

# -----------------------------------------------------------------------------
# 위에서 구한 상관계수의 검정 들어간다.
# 변수탐색기의 corr1 을 보면 p-value가 0.00000002842 임을 알 수 있고, 유의수준 0.01보다 작으므로 귀무가설을 기각한다. 따라서 상관계수는 0이 아니다.


# -----------------------------------------------------------------------------
# 회귀직선 절편과 기울기를 구하기 위해 회귀분석 패키지를 이용한다.
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
#선형회귀식에서 fit()은 (독립변수, 종속변수), 단 다변수일 경우를 대비해 독립변수는 [[]]로 쓰기.
reg.fit(df1[['export_revenue']], df1['import_revenue'])
slope1 = reg.coef_ # 회귀직선의 기울기 확인 , _(언더바)기호가 빠지면 안됨. intercept도 마찬가지. 
intercept1 = reg.intercept_

# -----------------------------------------------------------------------------
# 그래프 그려보기 , 이곳의 X에는 머신러닝 fit 처럼 독립변수에 2차원 리스트를 넣지 않는다.
import matplotlib.pyplot as plt
plt.scatter(df1['export_revenue'], df1['import_revenue'])
regline1 = slope1*df1['export_revenue'] +intercept1
plt.plot(df1['export_revenue'], regline1, color='red')
plt.title('y = {}*x + {}'.format(slope1, intercept1))
plt.show()

# 그린 그 회귀식이 산점도와는 100% 맞지 않지만 어떤 값을 갖는지 보고싶다면?
y_pred1 = reg.predict(df1[['export_revenue']])

# -----------------------------------------------------------------------------
# 책의 해답으로는 절편 b0 = -1.19, 기울기 b1 = 0.20으로 나왔다! 제대로 코딩한 것 맞다.

# -----------------------------------------------------------------------------
# 예제가 많이 적으니 윗 예제의 데이터로 분산분석 표까지 써보겠다.
import statsmodels.api as sm
# 이제 X를 df1['export_revenue']로 쓰기 너무 길고 번거로우니 치환 들어간다.
X = df1['export_revenue']
Y = df1['import_revenue']
results = sm.OLS(Y, sm.add_constant(X)).fit()
print(results.summary())











# 예제 9.2 (208p) , 이건 교재의 정답과 달랐음.... 교재는 기울 -142.9, 절편 1828.7 (지금보니 교재 오타인듯 하다. 해설과 보기의 수치가 다름.)
# 산점도를 그리고 이 상관도 위에 적합한 하나의 회귀직선을 그려라.

df2 = pd.DataFrame({'X':[1,2,3,6,8,10], 'Y':[1800,1400,1300,1000,600,500]})
corr2 = stats.pearsonr(df2['X'], df2['Y'])
reg.fit(df2[['X']],df2['Y'])
slope2 = reg.coef_
intercept2 = reg.intercept_
y_pred2 = reg.predict(df2[['X']])


