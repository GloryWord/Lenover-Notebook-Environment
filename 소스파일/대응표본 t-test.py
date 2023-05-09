# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 13:55:33 2022

@author: 이종웅
"""

import scipy as sp
from scipy import stats
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

df_paired_ttest = pd.read_csv('C:/Python_Data_Science_Statical_Study_example_source/part_03/data/ttest/tvads.csv')

print(df_paired_ttest.head(),"\n")

before = np.array(df_paired_ttest['before'])
after = np.array(df_paired_ttest['after'])

# 대응 표본 t 검정
paired_ttest_result = stats.ttest_rel(after, before)
# ttest_rel() 이 대응 표본 t-검정 함수인듯
print("Statistic(t-value) : %.3f p-value : %.28f" % paired_ttest_result)

plt.figure(figsize=(16,12))

ax1 = plt.subplot(221)
ax1 = sns.distplot(before, kde=False, fit = stats.gamma, label='before', color='blue')
ax1.set(xlabel="The number of customers", title="TV ads, After")
plt.legend()

ax2 = plt.subplot(222)
ax2 = sns.distplot(after, kde=False, fit = stats.gamma, label='after', color='red')
plt.legend()

# figsize : x축 y축 길이 설정 (단위 : inch)
# subplot() : 한 번에 여러 그래프를 같이 그리는 것. 221의 의미는 2,1,1
# displot 에서 fit=gamma 라고 되어있는데 fit=norm 해도 결과는 똑같다. 왜 gamma 인지 이유는 모르겠다.
plt.show()

# 함께 보여보기.
ax3 = plt.subplots()
ax3 = sns.distplot(before, kde=False, fit = stats.gamma, label='before', color='blue')
ax3 = sns.distplot(after, kde=False, fit = stats.gamma, label='after', color='red')
ax3.set(xlabel = "The number of cistomers", title = "TV ads, Before vs After")

plt.legend()
plt.show()