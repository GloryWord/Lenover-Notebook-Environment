# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 23:50:49 2022

@author: 이종웅
"""

import scipy as sp
from scipy import stats
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

df_independent_ttest = pd.read_csv('C:/Python_Data_Science_Statical_Study_example_source/part_03/data/ttest/promotion.csv')

print(df_independent_ttest.head())

print(df_independent_ttest[df_independent_ttest['promotion']=='NO'][['profit']].describe(),"\n")
print(df_independent_ttest[df_independent_ttest['promotion']=='YES'][['profit']].describe(),"\n")
print()
# DataFrame[][] 에 길게 채운 것 뿐이다.
# describe() 는 자료를 요약해준다. 갯수, 평균값, 표준편차 등등

# 프로모션 했을때, 안 했을때 차이점을 본다.
arr_yes = df_independent_ttest[df_independent_ttest['promotion']=='YES'][['profit']]
arr_no = df_independent_ttest[df_independent_ttest['promotion']=='NO'][['profit']]

# 등분산 검정 레빈(Levene)
# yes와 no DataFrame 2차원인데, levene 함수는 1차원만 가능 하므로
# DataFrame -> Serise로 바꿔준다. 
arr_yes = arr_yes.squeeze()
arr_no = arr_no.squeeze()
levene = stats.levene(arr_yes, arr_no)
print("levene result(F) : %.3f \np-value : %.3f" % (levene),"\n")

# 등분산 검정 플리그너(Fligner)
fligner = stats.fligner(arr_yes,arr_no)
print("fligner result(F) : %.3f \np-value : %.3f" % (fligner),"\n")

# 등분산 검정 바틀렛(Bartlett)
bartlett = stats.bartlett(arr_yes,arr_no)
print("bartlett result(F) : %.3f \np-value : %.3f" % (bartlett),"\n")

# 독립 표본 t 검정.
# equal_var = True 옵션운 분산이 같다. 이게 맞아? 묻는 옵션.
# ttest_ind() 인수에 등분산을 비교할 yes와 no 배열이 들어감.
ind_ttest_result = stats.ttest_ind(arr_no, arr_yes, equal_var = True)
print("Statisic(t-value: %.3f p-value : %.28f" % ind_ttest_result)

# "등분산에 대한" p-value가 0.05 보다 크므로 (각각0.612, 0.672, 0.739 이므로) 등분산 만족
# 이때의 "귀무가설은 두 집단의 분산이 동일하다" 이다. 기각역 밖이라 기각 못하고 귀무가설 맞음.
# "위 판단에 대한" p-value는 0.05보다 훨씬 작으므로 유의미하다. (기각역 안에 있다.)
