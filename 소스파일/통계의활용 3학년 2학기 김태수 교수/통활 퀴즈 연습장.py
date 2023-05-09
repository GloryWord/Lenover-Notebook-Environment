# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 21:51:36 2022

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


posterior_A_samples = [65,87,73,75]
posterior_B_samples = [75,69,83,81,72]
posterior_C_samples = [59,78,67,68]
posterior_D_samples = [87, 84, 81]




## Stats Model을 활용한 방법1. 
F, p = stats.f_oneway(posterior_A_samples, posterior_B_samples, posterior_C_samples, posterior_D_samples)
print( 'F-Ratio: {}'.format(F)
    , 'p-value:{}'.format(p)
     , sep = '\n')


result = sm.stats.anova_lm(posterior_A_samples, posterior_B_samples, posterior_C_samples, posterior_D_samples)
print(result)

# https://data-marketing-bk.tistory.com/entry/Python%EC%9D%B4%EC%9B%90%EB%B6%84%EC%82%B0%EB%B6%84%EC%84%9DTwo-way-ANOVA-%EC%BD%94%EB%93%9C%EB%B6%80%ED%84%B0-%EA%B2%B0%EA%B3%BC-%ED%95%B4%EC%84%9D-%EA%B0%80%EC%9D%B4%EB%93%9C?category=855602

# 위 주소는 이원분산분석을 실행하기 위해 데이터를 전처리하는 중요한 과정이 들어가 있다.







