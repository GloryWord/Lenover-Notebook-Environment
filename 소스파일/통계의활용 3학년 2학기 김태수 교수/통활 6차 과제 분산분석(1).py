# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:14:23 2022

@author: Leejongwoong
"""

import pandas as pd
import math
import numpy as np
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_excel("C:/Python_Data_Science_Statical_Study_example_source/Seoultech_University/Statistic_practice/1.adhd.xls")

# 모든 열의 요소 크기가 같으므로 len(df['D0']) 하나로 쭉 가도 좋다.
D0_sum_ex = sum(df['D0']) / len(df['D0'])
D15_sum_ex =sum(df['D15']) / len(df['D0'])
D30_sum_ex = sum(df['D30']) / len(df['D0'])
D60_sum_ex =sum(df['D60']) / len(df['D0'])

Total_sum_ex = ( sum(df['D0']) + sum(df['D15']) + sum(df['D30']) + sum(df['D60']) ) / (4*len(df['D0']))

TTS = 0 # Total Sum of Square
#for i in df:
#    for j in range(24):
#        TTS = df.loc[i][j] - Total_sum_ex  
print(stats.pearsonr(df['D0'],df['D60']))

leafs = [x%10 for x in df['D0']] ## 잎
stems = [x//10 for x in df['D0']] ## 줄기
fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white')
plt.stem(stems, leafs,use_line_collection=True) ## 줄기 잎 그림
plt.show()

import seaborn as sns
sns.boxplot(x="D0", y="D15",data=df)
sns.boxplot(x="D0", y="D30",data=df)
sns.boxplot(x="D0", y="D60",data=df)
sns.boxplot(x="D15", y="D30",data=df)
sns.boxplot(x="D15", y="D60",data=df)
sns.boxplot(x="D30", y="D60",data=df)
plt.show()
#print(len(df))
#print(len(df['D0']))
#D0_sum_ex_square = D0_sum_ex**2
#D15_sum_ex_square = D15_sum_ex**2
#D30_sum_ex_square = D30_sum_ex**2
#D60_sum_ex_square = D60_sum_ex**2

#SStr = 24 * ( D0_sum_ex_square + D15_sum_ex_square + D30_sum_ex_square + D60_sum_ex_square )

#SSE = TTS - SStr
#MStr = SStr / len(df)
#F_statistic = MStr / MSE
# print(len(df.columns))