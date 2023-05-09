# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:45:24 2022

@author: Leejongwoong
"""

import pandas as pd
import math
import numpy as np
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Python_Data_Science_Statical_Study_example_source/Seoultech_University/Statistic_practice/1.설탕소비.csv                                                               ")


df1 = df[(df['year']>=1822) & (df['year']<=2005)]
df2 = df[(df['year']>=1822) & (df['year']<=1930)]
df3 = df[(df['year']>=1931) & (df['year']<=1960)]
df4 = df[(df['year']>=1961) & (df['year']<=2005)]

plt.scatter(df1.year, df1.sugar_consum)
plt.title('1822~2005')
plt.xlabel('year')
plt.ylabel('sugar_consum')
plt.show()

plt.scatter(df2.year, df2.sugar_consum)
plt.title('1822~1930')
plt.xlabel('year')
plt.ylabel('sugar_consum')
plt.show()

plt.scatter(df3.year, df3.sugar_consum)
plt.title('1931~1960')
plt.xlabel('year')
plt.ylabel('sugar_consum')
plt.show()

plt.scatter(df4.year, df4.sugar_consum)
plt.title('1961~2005')
plt.xlabel('year')
plt.ylabel('sugar_consum')
plt.show()

X1 = df1['year']
Y1 = df1['sugar_consum']
results = sm.OLS(Y1, sm.add_constant(X1)).fit()
print(results.summary())

X2 = df2['year']
Y2 = df2['sugar_consum']
results = sm.OLS(Y2, sm.add_constant(X2)).fit()
print(results.summary())

X3 = df3['year']
Y3 = df3['sugar_consum']
results = sm.OLS(Y3, sm.add_constant(X3)).fit()
print(results.summary())

X4 = df4['year']
Y4 = df4['sugar_consum']
results = sm.OLS(Y4, sm.add_constant(X4)).fit()
print(results.summary())

corr1 = stats.pearsonr(df1['year'], df1['sugar_consum'])
corr2 = stats.pearsonr(df2['year'], df2['sugar_consum']) 
corr3 = stats.pearsonr(df3['year'], df3['sugar_consum']) 
corr4 = stats.pearsonr(df4['year'], df4['sugar_consum']) 