# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 20:29:33 2022

@author: 이종웅
"""

import numpy as np
import pandas as pd

from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

import sklearn.linear_model
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Python_Data_Science_Statical_Study_example_source/Kaggle_Data/archive/kc_house_data.csv')

# 컴퓨터 사이언스에서는 index 0부터 시작인데 1부터 시작하기 위한 작업을 하나보다. (교재에서)

df.index += 1 # index에 이런 사칙연산이 가능하구나.

# 결측치가 있는지 확인하기
print(df.isnull().sum())

# info()를 통해 컬럼의 수와 int, float와 같은 숫자형인지 object 타입인지 조사하는 탐색적 데이터 분석이 가능하다.
print(df.info())

# describe()를 통해 기술 통계값을 확인할 수 있다. 중간값, 결측치, 왜도, 첨도는 안나오므로 직접 구해야하는 듯.
# 이 과정에서 엄청난 데이터 핸들링 스킬이 들어가니까, 그림으로도 도식화 해보고 복습하자.
df_stats = df.describe().T # T는 t로도 쓸 수 있으며 transpose (전치)임.

skew_results = []
kurtosis_results = []
null_results = []
median_results = []

for idx, val in enumerate(df_stats.index):
    median_results.append(df[val].median())
    skew_results.append(df[val].skew())
    kurtosis_results.append(df[val].kurtosis())
    null_results.append(df[val].isnull().sum())
    
df_stats['median'] = median_results
df_stats['missing'] = null_results
df_stats['skewness'] = skew_results
df_stats['kurtosis'] = kurtosis_results

print("----- df 출력(column 수가 많아서 여러 줄로 출력된다.) -----\n")
print(df)
print()
print()
print("----- df_stats 시작 -----\n")
print(df_stats)

# 종속 변수인 price를 살펴 보는데, 왜도가 왼쪽으로 치우쳐 있는 것을 확인할 수 있다. (양수 왜도)
# 그래서 자연로그를 활용하여 분포를 조정할 수 있다.
plt.xlabel('Value')
plt.ylabel('Numbers of x value')
print(df['price'].hist())


print(np.log(df['price']).hist())

print("자연로그를 취한 price의 왜도 값 (0~1의 값이 나온다.) : ",np.log(df['price']).skew())

# 이제 종속 변수 price의 분포가 정규분포 형태를 띄는 것을 확인 했다.
# 다음은 데이터를 타입별로 분류하자. 전처리 과정이다.

def separate_dtype(df):
    df_obj = df.select_dtypes(include=['object'])
    df_numr = df.select_dtypes(include = ['int64', 'float64'])
    return[df_obj,df_numr]
(df_obj,df_numr) = separate_dtype(df)
print()
print(df_obj.head())
