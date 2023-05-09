# -*- coding: utf-8 -*-
"""
Created on Sat May 14 13:09:32 2022

@author: 이종웅
"""

# DO NOT CHANGE THIS PART
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%matplotlib inline

data=pd.read_csv('https://drive.google.com/uc?export=download&id=1Yo_VK8ntbGWe6Z8cEKNrWINUrpxqMc0Z')

trn,val=train_test_split(data,test_size=0.2,random_state=78)


# 3-(2)
# 종속변수를 0,1로만 이루어지도록 하기 위해 전처리
def binary(x):
    if x > 0:
        x=1
        return x
    else:
        x=0
        return x
    
pre_trn_y = trn["area"].apply(binary)
pre_val_y = val["area"].apply(binary)

# area라는 target변수와 설명변수가 아닌 것 떼어내기.
trn_X = trn.drop(['month','day','area'], axis=1)
val_X = val.drop(['month','day','area'], axis=1)


# 느낀점. 정확도를 평가하는데, y값과 예측y값을 비교하는 것은 from sklearn.metrics import accuracy_score를 해야 하는데, 교수님은 그걸 허용하지 않았기 때문에, knn_clf.score(val_X, pre_val_y) 같이 직접 검증용 x와 타겟을 입력하여 score를 얻어냈다.


# k가 1~11까지 돌려서 필요한 k값만 뽑아 쓰면 됨.
result = pd.DataFrame(columns=['Accuracy'])
for k in range(1,12):
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(trn_X, pre_trn_y)
    y_pred = knn_clf.predict(val_X)
    result.loc[k] = knn_clf.score(val_X, pre_val_y)

print(result)

# 3-(3)

# 20개의 이웃이 필요하므로 k=20 으로 설정, nn은 nearst neighbor 이다. 근접이웃.

knn_clf = KNeighborsClassifier(n_neighbors=20)
knn_clf.fit(trn_X, pre_trn_y)
nn = knn_clf.kneighbors(val_X) # nn[1][0~19]가 첫번째 샘플의 이웃들의 인덱스.


val_nn_RH = []
val_nn_wind = []
val_nn_index = nn[1][0] # 검증용 데이터의 첫번째 샘플의 이웃들의 인덱스 모아놓은 배열을 줬다.


# 목표 : 전체 데이터의 RH열 데이터중에서 인덱스가 nn[1][0~19]인 열만 추출해서 val_nn_RH에 옮기고 그 데이터의 수는 val_X의 첫 샘플포함해서 21이 되는 것이다.
# 목표 : 전체 데이터의 wind열 데이터중에서 인덱스가 nn[1][0~19]인 열만 추출해서 val_nn_wind에 옮기고 그 데이터의 수는 21이 되는 것이다.


for k in range(0,20):
    val_nn_RH.append(data.loc[val_nn_index[k],"RH"]) # 검증용 데이터 첫번째 샘플녀석 기준으로 이웃들의 RH 데이터를 가져온다. 
for k in range(0,20):
    val_nn_wind.append(data.loc[val_nn_index[k],"wind"]) # 검증용 데이터 첫번째 샘플녀석 기준으로 이웃들의 wind 데이터를 가져온다.

# 마지막으로 검증용 데이터 해당 샘플의 RH, wind 데이터도 추가.
val_nn_RH.append(val_X.iloc[0,5])
val_nn_wind.append(val_X.iloc[0,6])

# 이제 산점도를 그립니다.
plt.title("RH, wind")
plt.xlabel('RH')
plt.ylabel('wind')
plt.ylim((0,50))
plt.scatter(val_nn_RH,val_nn_wind,color="g",marker="x")





#x = np.random.rand(10)
#y = np.random.rand(10)
#z = np.sqrt(x**2 + y**2)
#plt.scatter(x, y, s=80, c=z, marker=(5, 2))



# 3-(4)
# data를 Z-score 표준화 시킨 pre_data 생성.
pre_data = pd.DataFrame(columns=['FFMC', 'DMC', 'DC', 'ISI','temp','RH','wind','rain','area'])
pre_data['FFMC']=(data["FFMC"]-data["FFMC"].mean())/(data["FFMC"].std())
pre_data['DMC']=(data["DMC"]-data["DMC"].mean())/(data["DMC"].std())
pre_data['DC']=(data["DC"]-data["DC"].mean())/(data["DC"].std())
pre_data['ISI']=(data["ISI"]-data["ISI"].mean())/(data["ISI"].std())
pre_data['temp']=(data["temp"]-data["temp"].mean())/(data["temp"].std())
pre_data['RH']=(data["RH"]-data["RH"].mean())/(data["RH"].std())
pre_data['wind']=(data["wind"]-data["wind"].mean())/(data["wind"].std())
pre_data['rain']=(data["rain"]-data["rain"].mean())/(data["rain"].std())
pre_data['area']=data['area']

z_trn,z_val=train_test_split(pre_data,test_size=0.2,random_state=78)

# 종속변수를 0,1로만 이루어지도록 하기 위해 전처리
def binary(x):
    if x > 0:
        x=1
        return x
    else:
        x=0
        return x
    
pre_z_trn_y = z_trn["area"].apply(binary)
pre_z_val_y = z_val["area"].apply(binary)

# area라는 target변수와 설명변수가 아닌 것 떼어내기.
z_trn_X = z_trn.drop(['area'], axis=1)
z_val_X = z_val.drop(['area'], axis=1)

# k가 1~11까지 돌려서 필요한 k값만 뽑아 쓰기.
z_result = pd.DataFrame(columns=['Accuracy'])
for k in range(1,12):
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(z_trn_X, pre_z_trn_y)
    y_pred = knn_clf.predict(z_val_X)
    z_result.loc[k] = knn_clf.score(z_val_X, pre_z_val_y)
print(z_result)

# 3-(5)
knn_clf = KNeighborsClassifier(n_neighbors=20)
knn_clf.fit(z_trn_X, pre_z_trn_y)
nn = knn_clf.kneighbors(z_val_X)

z_val_nn_RH = []
z_val_nn_wind = []
z_val_nn_index = nn[1][0]

for k in range(0,20):
    z_val_nn_RH.append(pre_data.loc[z_val_nn_index[k],"RH"]) 
for k in range(0,20):
    z_val_nn_wind.append(pre_data.loc[z_val_nn_index[k],"wind"]) 
z_val_nn_RH.append(z_val_X.iloc[0,5])
z_val_nn_wind.append(z_val_X.iloc[0,6])


plt.title("RH, wind (Z-score)")
plt.xlabel('RH')
plt.ylabel('wind')
plt.xlim((-2,2))
plt.ylim((-2,2))
plt.scatter(z_val_nn_RH,z_val_nn_wind)


