# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 12:14:42 2022

@author: 이종웅
"""

# 데이터 핸들링 툴
import pandas as pd
import numpy as np

# 통계 툴
import scipy as sp
from scipy import stats
import statsmodels.api as sm # API는 프레임워크나 라이브러리 개념이다.
from statsmodels.formula.api import ols
# OLS란 Ordinary Least Squares(최소제곱법)

# 시각화 패키지
from matplotlib import pyplot as plt
import seaborn as sns

# 경고 무시
import warnings
warnings.filterwarnings('ignore')

df_f_oneway = pd.read_csv('C:/Python_Data_Science_Statical_Study_example_source/part_03/data/anova/factory.csv')

# 데이터 셋 정보 확인
print()
print(df_f_oneway.info(),"\n")

# 결측치 확인
print(df_f_oneway.isnull().sum(),"\n")

# 업체별 데이터 개수 확인
print("---업체별 데이터 개수 확인---")
print(df_f_oneway['type'].value_counts(),"\n")

# 기술 통계량 확인
print("---기술 통계량 확인---")
print(df_f_oneway.describe(),"\n")

# 왜도 계산 함수
def skew(x):
    return stats.skew(x)

# 첨도 계산 함수
def kurtosis(x):
    return stats.kurtosis(x)

# groupby는 원하는 열 내부에서의 여러가지 값들을 따질수 있도록 인수에 들어간 행의 데이터들을 싹 모아준다. 그리고 여기의 인수는 큰 따옴표를 쓴다...
# type 별로 불량률을 출력해 봤더니 불량률 평균값은 type 4가 가장 높다. 3번이 가장 낮다. 
df_desc_stat = df_f_oneway.groupby("type")['defRate'].describe()


# 기존 데이터 프레임에 중간값, 왜도, 첨도, 결측값도 추가하는 코드.
skew_results = []
kurtosis_results = []
null_results = []

for i in range(1,5):
    skew_results.append(skew(df_f_oneway[df_f_oneway['type']==i]['defRate']))
    kurtosis_results.append(kurtosis(df_f_oneway[df_f_oneway['type']==i]['defRate']))
    null_results.append(df_f_oneway[df_f_oneway['type']==i]['defRate'].isnull().sum())

df_desc_stat['median']=df_f_oneway.groupby("type")['defRate'].median()
df_desc_stat['skew']=skew_results
df_desc_stat['kurtosis']=kurtosis_results
df_desc_stat['missing']=null_results

# head로 최종 요약하여 보아도, type 3이 불량률의 중간값이 가장 작다.
print(df_desc_stat.head())

type1 = np.array(df_f_oneway[df_f_oneway['type']==1]['defRate'])
type2 = np.array(df_f_oneway[df_f_oneway['type']==2]['defRate'])
type3 = np.array(df_f_oneway[df_f_oneway['type']==3]['defRate'])
type4 = np.array(df_f_oneway[df_f_oneway['type']==4]['defRate'])

figure,((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(16,12)

sns.distplot(type1, norm_hist=False, kde=False, label='type1',rug=True, color='blue', ax=ax1)
sns.distplot(type2, norm_hist=False, kde=False, label='type2',rug=True, color='red', ax=ax2)
sns.distplot(type3, norm_hist=False, kde=False, label='type3',rug=True, color='green', ax=ax3)
sns.distplot(type4, norm_hist=False, kde=False, label='type4',rug=True, color='orange', ax=ax4)

ax1.set(ylabel='defRate', title="type1")
ax2.set(ylabel='defRate', title="type2")
ax3.set(ylabel='defRate', title="type3")
ax4.set(ylabel='defRate', title="type4")

plt.show()