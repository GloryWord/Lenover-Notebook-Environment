# (11) M1 모델을 기준으로 (1)에서 선택된 변수들에 대해서만, 개별 설명변수를 x축으로 하고 y축을 잔차(residual)로 하는 산포도(scatter plot)을 그리시오. 
# 산포도를 보고 알 수 있는 사항에 대해서 기술하시오. 

# alcohol,  volateile acidity,  sulphates,  citric acid,  total sulfur dioxide을 선택.
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import t, f, probplot
from scipy import stats
%matplotlib inline
data=pd.read_csv('https://drive.google.com/uc?export=download&id=1Bs6z1GSoPo2ZPr5jL2qDjRghYcMUOHbS')
reg = LinearRegression()
x = data[['volatile acidity', 'citric acid', 'total sulfur dioxide', 'sulphates', 'alcohol']]
y = data['quality']
reg.fit(x, y)
y_pred = reg.predict(x)

error = y-y_pred
plt.title('alcohol , error')
plt.xlabel('alcohol')
plt.ylabel('error')
plt.scatter(data[['alcohol']], error)
plt.show()

plt.hist(error)
plt.show()
stats.probplot(error, dist - 'norm', plot-plt)

import scipy.stats as stats
stats.probplot(error, plot=plt)
#plt.scatter(x[:0], error)



