# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import scipy as sp
from scipy import stats
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# 경고 메시지 무시하기
import warnings
warnings.filterwarnings('ignore') 

# 감자칩의 무게가 줄었는지 확인하는 검정 예제

potato_chip = np.array([89.03, 95.07, 88.26, 90.07, 90.6, 87, 87.67, 88.8, 90.46, 81.33])

# scipy의 ttest_lsamp 사용. 90은 귀무가설의 기댓값 중량 90g 를 뜻함. (제조업체에서 주장하는 자기네들의 중량 기댓값이 90g 이라는 귀무가설)

# 단일 t-test (one sample t-test)를 쓴 이유는, 단일 그룹의 평균이 어떤 상수와 같은지 검정하기 위함.
one_sample = stats.ttest_1samp(potato_chip, 90)

print("Statistic(t-value): %.3f p-value : %.4f'" % one_sample)

# kde(Kerner Density Estimation): 구해진 히스토그램을 정규화한 뒤 확률 밀도 함수로 사용한다.
# 커널 밀도(kernel density)는 커널이라는 함수를 겹치는 방법으로 히스토그램보다 부드러운 형태의 분포 곡선을 보여주는 방법이다.
ax = sns.distplot(potato_chip, kde=False, fit=stats.norm, label="potato_chip", color ='green', rug=True)
ax.set(xlabel= "The weight for the snack")
plt.legend()
plt.show()
