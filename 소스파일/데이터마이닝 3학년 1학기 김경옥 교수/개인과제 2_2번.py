# -*- coding: utf-8 -*-
"""
Created on Sat May 14 23:45:44 2022

@author: 이종웅
"""

# DO NOT CHANGE THIS PART
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, recall_score, precision_score

data=pd.read_csv('https://drive.google.com/uc?export=download&id=1ds25l6300RnG7MMzRmtrHVdlUXmVsUvi')


# 2-(1)
from sklearn.model_selection import train_test_split

trn, test=train_test_split(data,test_size=0.2, shuffle=True, random_state=0)

trn_X = trn.drop('target',axis=1)
trn_y= trn['target']
test_X = test.drop('target', axis=1)
test_y = test['target']

t1 = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_split=50, min_samples_leaf=25)

t1.fit(trn_X,trn_y)
y_pred = t1.predict(test_X)

print("정확도 : \n",t1.score(test_X,test_y))
print("Class 1의 recall : \n", recall_score(test_y, y_pred, pos_label=1))
print("Class 0의 recall : \n", recall_score(test_y, y_pred, pos_label=0))

#recall_score(test_y, y_pred, pos_label=0)
print("Class 1의 precision : \n",precision_score(test_y, y_pred, pos_label=1))
print("Class 0의 precision : \n ",precision_score(test_y, y_pred, pos_label=0))


# 2-(3)

tree.plot_tree(t1)

feature_names = data.columns[:4]
target_names = str(data['target'].unique().tolist())
fig = plt.figure(dpi = 800)
tree.plot_tree(t1,feature_names = feature_names, class_names = target_names,filled = True)


