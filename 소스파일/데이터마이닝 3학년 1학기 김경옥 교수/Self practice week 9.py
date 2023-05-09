# -*- coding: utf-8 -*-
"""
Created on Thu May  5 22:24:22 2022

@author: 이종웅
"""

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data
y= iris.target


Xtr, Xts, Ytr, Yts = train_test_split(X,y,test_size=0.2, shuffle=True)

len(X)
len(Xtr)
len(Xts)


knn_clf = KNeighborsClassifier(n_neighbors=3, metric='manhattan') 
knn_clf.fit(Xtr,Ytr) # train셋으로 훈련시킨 코드임.

y_pred = knn_clf.predict(Xts) # 위의 훈련의 결과를 바탕으로 test datasets을 넣어봐서 이렇게  분류 될 것이다라는 예측값.


nn = knn_clf.kneighbors(Xts) # 현재 k=3임(n_neighbors=3), 입력한 Xts 데이터와 가장 가까운 이웃 중 k개의 "거리"와 "인덱스"를 반환. 주어진 샘플에서 가장 가까운 이웃을 찾아주는 함수임. 반환값은 이웃까지의 거리와 이웃 샘플의 인덱스를 반환함.여기서의 index 1은 이웃 샘플의 인덱스를 k개 (3개 반환함.) 그래서 인덱스 들이 실제 데이터의 개수인 150을 넘지 않음.

knn_clf.score(Xts, Yts)

cov_mat = np.cov(Xts.T) # 공분산 계산
cov_mat.shape


knn_clf2 = KNeighborsClassifier(n_neighbors=5, metric='mahalanobis', metric_params={'V':cov_mat}, weights='distance') # 무게에 s 안붙이면 안되고 위에 중괄호 아니면 안됨. 마할라노비스 거리체계는 공분산 행렬이 필요함.
knn_clf2.fit(Xtr,Ytr)

y_pred2 = knn_clf2.predict(Xts)
nn2 = knn_clf2.kneighbors(Xts)

knn_clf2.score(Xts,Yts)



rnn_clf = RadiusNeighborsClassifier(radius=0.8)
rnn_clf.fit(Xtr,Ytr)

y_pred3 = rnn_clf.predict(Xts)
nn3 = rnn_clf.radius_neighbors(Xts) # 반경 내의 이웃 간 거리와 샘플 인덱스들이 출력됨.


len(nn3[1][0])# 행렬순서가 거리, 인덱스 순
len(nn3[1][2])
# 여기부턴 일단 바로 베껴적었다. 결과를 알아야 하니까. 

nn_size =[len(x) for x in nn3[1]]
nn_size

import matplotlib.pyplot as plt

np.random.seed(10)
Xtr = np.random.rand(200)*10
np.random.seed(20)
Ytr = np.sin(Xtr)*2 + np.random.normal(size=200)*0.5
plt.scatter(Xtr, Ytr)


Xts = np.linspace(0,10,50)



















from sklearn.preprocessing import scale
import pandas as pd
from sklearn.metrics import pairwise_distances

n=20
x = np.random.rand(n)
y = np.random.rand(n)*10

plt.scatter(x,y)
plt.xlim((0,10))
plt.ylim((0,10))


train = pd.DataFrame({'x':x, 'y':y})

d_test = pairwise_distances(Xtr.reshape((-1,1)))

point = [0.5,0.5]

train['dist']=pairwise_distances([point],train)[0]


k=5

sort_train = sort








import pandas as pd
























































from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


iris = datasets.load_iris()
X = iris.data
y= iris.target

cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target


Xtr, Xts, Ytr, Yts = train_test_split(X,y,test_size=0.2, shuffle=True)


knn_clf = KNeighborsClassifier()

result = pd.DataFrame(columns=['Accuracy','Recall','Precision', 'F1'])
for k in range(1,11):
    knn_clf.n_neighbors = k
    knn_clf.fit(Xtr,Ytr)
    y_pred = knn_clf.predict(Xts)
    result.loc[k] = [accuracy_score(Yts, y_pred), recall_score(Yts,y_pred), precision_score(Yts, y_pred), f1_score(Yts, y_pred)]






