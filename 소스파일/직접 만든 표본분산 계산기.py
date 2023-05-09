# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 21:02:43 2022

@author: Leejongwoong
"""

import pandas as pd
import math
import numpy as np
import scipy as sp
from scipy import stats
import statsmodels.api as sm

A = [50,70,70,40,90,70,60,50,80,60]
B = [60,60,80,50,80,60,60,60,80,70]

# 분산을 구하기 전 평균을 알아야 함.

mean_A = np.mean(A)
mean_B = np.mean(B)

# 표본분산은 결국 편차의 제곱 '합' 이기 때문에 합을 구해준다.
# def 로 함수 구현하면 되지만 일단 바빠서 그건 다음에

vsum = 0
for i in A:
    vsum = vsum + (i-mean_A)**2
sampled_variance_A = vsum / (len(A)-1) # 표본분산이 아니라면 -1 없애고 len(A).

# 아니 시발.... len(A)-1이 사칙연산 순서에 의해서 맨 마지막에 -1이 되어버렸다.
vsum = 0
for i in B:
    vsum = vsum + (i-mean_B)**2
sampled_variance_B = vsum / (len(B)-1)

# Numpy로 구한 분산
numpy_var_A = np.var(A)
numpy_var_B = np.var(B)
# 검정통계량 t 값
#t = (mean_A - 72)/(math.sqrt(sampled_variance_A/len(A)))


# 패키지로 t 검정 하기
from scipy.stats import ttest_1samp
#print()
#print("t검정 결과 : ",ttest_1samp(A, 72))