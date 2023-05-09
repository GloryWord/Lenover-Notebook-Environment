# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 22:22:06 2022

@author: 이종웅
"""

import pandas as pd
import math
import numpy as np
import scipy as sp
from scipy import stats
import statsmodels.api as sm


A=[291,283,292,301,327,285,319,325,332,313,322,300,302,284,291]
B=[286,267,278,312,268,263,287,278,288,259,274,273,243,259,299]


num_mean_A = np.mean(A)
num_mean_B = np.mean(B)
#print(num_mean)


# 분산
vsum = 0
for val in A:
    vsum = vsum + (val - num_mean_A)**2
variance_A = vsum / 14

vsum = 0
for val in B:
    vsum = vsum + (val - num_mean_B)**2
variance_B = vsum / 14
#print(variance_A)

F0 = variance_A/variance_B
Sp = (14*variance_A)+(14*variance_B)/28
root_Spp = math.sqrt(Sp)