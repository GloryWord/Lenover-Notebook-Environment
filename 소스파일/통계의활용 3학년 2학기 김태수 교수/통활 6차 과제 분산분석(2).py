# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 19:45:47 2022

@author: Leejongwoong
"""

import pandas as pd
import math
import numpy as np
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_excel("C:/Python_Data_Science_Statical_Study_example_source/Seoultech_University/Statistic_practice/2.비만의 편견.xls")

print(np.mean(df['QUALIFID']))
print(np.std(df['QUALIFID']))
print(stats.pearsonr(df['WEIGHT'],df['RELATE']))
passed_applicant = df[df['QUALIFID'] >= 7]

plt.scatter(df['WEIGHT'], df['RELATE'])
plt.show()

sm.qqplot(df['QUALIFID'], fit=True, line='45')

# 이원분류 분산분석 시작
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

model = ols('QUALIFID ~ WEIGHT * RELATE', df).fit()
print(anova_lm(model))

# 0~79 까지의 Qualified 평균을 구해보자
#print(np.mean(df[0:80]))