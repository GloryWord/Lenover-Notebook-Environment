# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:06:15 2022

@author: Leejongwoong
"""

import pandas as pd
import math
import numpy as np
import scipy as sp
from scipy import stats
import statsmodels.api as sm

df = pd.read_csv("C:/Users/이종웅/Rstudy/1.Air.csv")
print(stats.pearsonr(df['Ozone'],df['Solar.R']))
print(stats.pearsonr(df['Ozone'],df['Wind']))
print(stats.pearsonr(df['Ozone'],df['Temp']))
print(stats.pearsonr(df['Solar.R'],df['Wind']))
print(stats.pearsonr(df['Solar.R'],df['Temp']))
print(stats.pearsonr(df['Wind'],df['Temp']))