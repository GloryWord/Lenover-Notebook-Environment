# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:26:48 2022

@author: 이종웅
"""

from sklearn.naive_bayes import BernoulliNB, CategoricalNB, MultinomialNB, GaussianNB
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target



gausNB = GaussianNB()
gausNB.fit(X,y)

gausNB.theta_ # 붓꽃 데이터를 적합시킨 가우시안 모델의 세타가 나옴.
gausNB.sigma_ # 붓꽃 데이터를 적합시킨 가우시안 모델의 시그마가 나옴.

gausNB.class_prior_ # 각 클래스의 확률값을 보여줌. (sepal의 가로,세로, petal의 가로,세로) 총 4개임. 클래스 상으로는 0~3번 클래스로 인식 됨.
y_pred = gausNB.predict(X) # 분류 실행했음. (분류한 것은 예측값임. 실제 데이터가 아니므로.)
y_prob = gausNB.predict_proba(X) # 위에 분류한 (추정한)값이 맞을 확률.

np.sum(y_prob,axis=1) # 왜있는지 모름

gausNB.score(X,y) # 0.96
 