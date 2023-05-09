# -*- coding: utf-8 -*-
"""
Created on Sat May 14 20:20:03 2022

@author: 이종웅
"""

# DO NOT CHANGE THIS PART
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
#%matplotlib inline

data=pd.read_csv('https://drive.google.com/uc?export=download&id=1ZdJM0WQTaw3D2JRSNgV1KSyN2_-b0o2k', index_col=0)
trn, test=train_test_split(data,test_size=0.2, shuffle=True, random_state=11)

# 1-(1)

trn_X = trn.drop('y_target',axis=1)
trn_y= trn['y_target']
test_X = test.drop('y_target', axis=1)
test_y = test['y_target']

multiNB = MultinomialNB()
multiNB.fit(trn_X,trn_y)

y_pred = multiNB.predict(test_X)
y_true = test_y

result = [] #pd.DataFrame(columns=['Accuracy','Recall','Precision','F1'])

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
result = [accuracy_score(y_true,y_pred),recall_score(y_true,y_pred),precision_score(y_true,y_pred),f1_score(y_true,y_pred)]


result


# 1-(2)

from pandas import Series, DataFrame
pd.set_option('display.max_columns', 30)


# 우선 분류기를 통해 분류된 데이터에서 타겟값이 0인 것의 인덱스를 찾습니다.
cg_index = []
cg_index = np.where(y_pred == 0) # 타겟값이 0인 것의 인덱스를 찾기.
cg_index = list(np.where(y_pred == 0)) # 튜플은 수정이 안되므로 조작 용이하도록 리스트

cg_data_count = test_X.iloc[cg_index[0][0]] # Input data들에서 컴퓨터그래픽 범주값만 골라서 모든 요소들을(=단어별 나온 횟수) 더했음. 이것은 첫번째 컴퓨터그래픽 범 데이터.

for k in range(1, len(cg_index[0])):
    cg_data_count = cg_data_count + test_X.iloc[cg_index[0][k]]
    
print("컴퓨터 그래픽의 단어별 등장 빈도수 (내림차순)\n\n",cg_data_count.sort_values(ascending = False)) #내림차순으로 상위 10개 단어 출력

# ss = space science 를 뜻함.
ss_index = []
ss_index = np.where(y_pred == 1)
ss_index = list(np.where(y_pred == 1))
ss_data_count = test_X.iloc[cg_index[0][0]]

for k in range(1, len(ss_index[0])):
    ss_data_count = ss_data_count + test_X.iloc[ss_index[0][k]]
    
print("\n\n우주 과학의 단어별 등장 빈도수 (내림차순)\n\n",ss_data_count.sort_values(ascending = False))

#cg_data.append(test_X.iloc[cg_index[0][k]]) # 행 데이터가 뽑히긴 하지만 column형태로 뽑힘.
#pd.concat([df_1, Series_1], axis=1)

#A = {'num':['1','2','3','4','5']}
#B = {'num':['0','0','0','0','0']}
#dfA = pd.DataFrame(A)
#dfB = pd.DataFrame(B)
#for k in range(0,4):
    #dfB[k][num] = dfB[k][num] + dfA[k][num]  
#dfA['num']
#len(cg_index[0])
#cg_index[0][0]



