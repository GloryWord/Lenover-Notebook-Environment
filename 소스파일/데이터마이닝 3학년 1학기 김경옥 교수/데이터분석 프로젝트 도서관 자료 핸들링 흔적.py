temp3 = temp2.reset_index()
pre_data = temp3.drop('대출자의키',axis=1)
pre_data = temp2.reset_index() # 복잡한 대출자의 키를 간단한 인덱스로 변경
raw_data = pd.read_csv('C:/Users/이종웅/OneDrive - 서울과학기술대학교/서울과학기술대학교/서울과학기술대학교/데이터마이닝_김경옥/기말 과제/도서관 데이터/충청북도_행정간행물홈페이지_도서대여목록_20200630.csv')

# 저작자, 분류기호, 발행자, 서명, 대출자의키 뽑아내기
data = raw_data[['저작자', '분류기호', '발행자', '서명', '대출자의키']]

temp = data.groupby('대출자의키').count()
idx_num_1 = temp[temp['저작자']==1].index # 1권만 빌린 사람 제거
temp2 = temp.drop(idx_num_1) # 그 행은 삭제

pre_data = temp2.reset_index() # 복잡한 대출자의 키를 간단한 인덱스로 변경
raw_data = pd.read_csv('C:/Users/이종웅/OneDrive - 서울과학기술대학교/서울과학기술대학교/서울과학기술대학교/데이터마이닝_김경옥/기말 과제/도서관 데이터/충청북도_행정간행물홈페이지_도서대여목록_20200630.csv')
raw_data = pd.read_csv('C:/Users/이종웅/OneDrive - 서울과학기술대학교/서울과학기술대학교/서울과학기술대학교/데이터마이닝_김경옥/기말 과제/도서관 데이터/충청북도_행정간행물홈페이지_도서대여목록_20200630.csv', encoding ='utf-8')

## ---(Thu May 26 13:24:50 2022)---
import pandas as pd
import numpy as np

raw_data = pd.read_csv('C:/Users/이종웅/OneDrive - 서울과학기술대학교/서울과학기술대학교/서울과학기술대학교/데이터마이닝_김경옥/기말 과제/도서관 데이터/충청북도_행정간행물홈페이지_도서대여목록_20210601.csv')

# 저작자, 분류기호, 발행자, 서명, 대출자의키 뽑아내기
data = raw_data[['저작자', '분류기호', '발행자', '서명', '대출자의키']]

temp = data.groupby('대출자의키').count()
idx_num_1 = temp[temp['저작자']==1].index # 1권만 빌린 사람 제거
temp2 = temp.drop(idx_num_1) # 그 행은 삭제

pre_data = temp2.reset_index() # 복잡한 대출자의 키를 간단한 인덱스로 변경
user_index = temp2.reset_index() # 복잡한 대출자의 키를 간단한 인덱스로 변경
data['분류기호']
data[['분류기호'],0]
data[['분류기호']0]
data[[2],0]
data[[2,1]]
def genre_classification(x):
def genre_classification(x):
    if x == 0:
        x = '사회'
        return x
    elif x >= 1 and x <= 3.99:
        x = '교양' 
        return x
    elif x >= 4 and x <= 6:
        x = '수험서' 
        return x
    elif x >= 20 and x <= 30:
        x = '문헌정보' 
        return x
    elif x >= 31 and x <= 41:
        x = '교양' 
        return x
    elif x >= 80 and x <= 99:
        x = '만화' 
        return x
    elif x >= 100 and x <= 167:
        x = '철학' 
        return x
    elif x >= 180 and x <= 199:
        x = '심리학' 
        return x
    elif x >= 200 and x <= 299:
        x = '종교' 
        return x
    elif x >= 300 and x <= 310:
        x = '통계학' 
        return x
    elif x >= 311 and x <= 325.19:
        x = '경제,경영' 
        return x
    elif x >= 325.2 and x <= 326.79:
        x = '자기계발' 
        return x
    elif x >= 327 and x <= 329:
        x = '경제,경영' 
        return x
    elif x >= 330 and x <= 390:
        x = '사회과학' 
        return x
    elif x >= 400 and x <= 499:
        x = '순수과학' 
        return x
    elif x >= 500 and x <= 599:
        x = '기술과학' 
        return x
    elif x >= 600 and x <= 699:
        x = '예술' 
        return x
    elif x >= 700 and x <= 799:
        x = '언어'
        return x
    elif x >= 800 and x <= 899:
        x = '문학' 
        return x
    elif x >= 900 and x <= 998:
        x = '역사' 
        return x
    elif x >= 999 and x <= 69999:
        x = '정부행정문서' 
        return x
    else:
        x = '분류오류'
        return x
pre_data = data["분류기호"].apply(genre_classification)
def genre_classification(x):
    if x == 0:
        x = '사회'
        return x
    elif x >= 1 and x <= 3.99:
        x = '교양' 
        return x
    elif x >= 4 and x <= 6:
        x = '수험서' 
        return x
    elif x >= 20 and x <= 30:
        x = '문헌정보' 
        return x
    elif x >= 31 and x <= 41:
        x = '교양' 
        return x
    elif x >= 80 and x <= 99:
        x = '만화' 
        return x
    elif x >= 100 and x <= 167:
        x = '철학' 
        return x
    elif x >= 180 and x <= 199:
        x = '심리학' 
        return x
    elif x >= 200 and x <= 299:
        x = '종교' 
        return x
    elif x >= 300 and x <= 310:
        x = '통계학' 
        return x
    elif x >= 311 and x <= 325.19:
        x = '경제,경영' 
        return x
    elif x >= 325.2 and x <= 326.79:
        x = '자기계발' 
        return x
    elif x >= 327 and x <= 329:
        x = '경제,경영' 
        return x
    elif x >= 330 and x <= 390:
        x = '사회과학' 
        return x
    elif x >= 400 and x <= 499:
        x = '순수과학' 
        return x
    elif x >= 500 and x <= 599:
        x = '기술과학' 
        return x
    elif x >= 600 and x <= 699:
        x = '예술' 
        return x
    elif x >= 700 and x <= 799:
        x = '언어'
        return x
    elif x >= 800 and x <= 899:
        x = '문학' 
        return x
    elif x >= 900 and x <= 998:
        x = '역사' 
        return x
    elif x >= 999 and x <= 69999:
        x = '정부행정문서' 
        return x
    else:
        return x
pre_data = data["분류기호"].apply(genre_classification)
def genre_classification(x):
    if x == 0:
        x = '사회'
        return x
    elif x >= 1 and x <= 3.99:
        x = '교양' 
        return x
    elif x >= 4 and x <= 6:
        x = '수험서' 
        return x
    elif x >= 20 and x <= 30:
        x = '문헌정보' 
        return x
    elif x >= 31 and x <= 41:
        x = '교양' 
        return x
    elif x >= 80 and x <= 99:
        x = '만화' 
        return x
    elif x >= 100 and x <= 167:
        x = '철학' 
        return x
    elif x >= 180 and x < 200:
        x = '심리학' 
        return x
    elif x >= 200 and x < 300:
        x = '종교' 
        return x
    elif x >= 300 and x <= 310:
        x = '통계학' 
        return x
    elif x >= 311 and x <= 325.19:
        x = '경제,경영' 
        return x
    elif x >= 325.2 and x <= 326.79:
        x = '자기계발' 
        return x
    elif x >= 327 and x <= 329:
        x = '경제,경영' 
        return x
    elif x >= 330 and x <= 390:
        x = '사회과학' 
        return x
    elif x >= 400 and x <= 499:
        x = '순수과학' 
        return x
    elif x >= 500 and x <= 599:
        x = '기술과학' 
        return x
    elif x >= 600 and x <= 699:
        x = '예술' 
        return x
    elif x >= 700 and x <= 799:
        x = '언어'
        return x
    elif x >= 800 and x <= 899:
        x = '문학' 
        return x
    elif x >= 900 and x <= 999:
        x = '역사' 
        return x
    elif x >= 1000 and x <= 69999:
        x = '정부행정문서' 
        return x
    else:
        return x
pre_data = data["분류기호"].apply(genre_classification)
def genre_classification(x):
    if x == 0:
        x = '사회'
        return x
    elif x >= 1 and x <= 3.99:
        x = '교양' 
        return x
    elif x >= 4 and x < 7:
        x = '수험서' 
        return x
    elif x >= 7 and x <= 30:
        x = '문헌정보' 
        return x
    elif x >= 31 and x <= 41:
        x = '교양' 
        return x
    elif x >= 80 and x <= 99:
        x = '만화' 
        return x
    elif x >= 100 and x <= 167:
        x = '철학' 
        return x
    elif x >= 180 and x < 200:
        x = '심리학' 
        return x
    elif x >= 200 and x < 300:
        x = '종교' 
        return x
    elif x >= 300 and x <= 310:
        x = '통계학' 
        return x
    elif x >= 311 and x <= 325.19:
        x = '경제,경영' 
        return x
    elif x >= 325.2 and x <= 326.79:
        x = '자기계발' 
        return x
    elif x >= 327 and x <= 329:
        x = '경제,경영' 
        return x
    elif x >= 330 and x <= 390:
        x = '사회과학' 
        return x
    elif x >= 400 and x <= 499:
        x = '순수과학' 
        return x
    elif x >= 500 and x <= 599:
        x = '기술과학' 
        return x
    elif x >= 600 and x <= 699:
        x = '예술' 
        return x
    elif x >= 700 and x <= 799:
        x = '언어'
        return x
    elif x >= 800 and x <= 899:
        x = '문학' 
        return x
    elif x >= 900 and x <= 999:
        x = '역사' 
        return x
    elif x >= 1000 and x <= 69999:
        x = '정부행정문서' 
        return x
    else:
        return x
pre_data = data["분류기호"].apply(genre_classification)
def genre_classification(x):
    if x == 0:
        x = '사회'
        return x
    elif x >= 1 and x <= 3.99:
        x = '교양' 
        return x
    elif x >= 4 and x < 7:
        x = '수험서' 
        return x
    elif x >= 7 and x <= 30:
        x = '문헌정보' 
        return x
    elif x >= 31 and x <= 41:
        x = '교양' 
        return x
    elif x >= 80 and x <= 99:
        x = '만화' 
        return x
    elif x >= 100 and x <= 167:
        x = '철학' 
        return x
    elif x >= 180 and x < 200:
        x = '심리학' 
        return x
    elif x >= 200 and x < 300:
        x = '종교' 
        return x
    elif x >= 300 and x <= 310:
        x = '통계학' 
        return x
    elif x >= 311 and x <= 325.19:
        x = '경제,경영' 
        return x
    elif x >= 325.2 and x <= 326.79:
        x = '자기계발' 
        return x
    elif x >= 327 and x < 330:
        x = '경제,경영' 
        return x
    elif x >= 330 and x <= 390:
        x = '사회과학' 
        return x
    elif x >= 400 and x <= 499:
        x = '순수과학' 
        return x
    elif x >= 500 and x <= 599:
        x = '기술과학' 
        return x
    elif x >= 600 and x <= 699:
        x = '예술' 
        return x
    elif x >= 700 and x <= 799:
        x = '언어'
        return x
    elif x >= 800 and x <= 899:
        x = '문학' 
        return x
    elif x >= 900 and x <= 999:
        x = '역사' 
        return x
    elif x >= 1000 and x <= 69999:
        x = '정부행정문서' 
        return x
    else:
        return x
pre_data = data["분류기호"].apply(genre_classification)
genre_classification = data["분류기호"].apply(genre_classification)
data.combine(genre_classification, take_smaller)
data.combine(genre_classification)
data.combine(genre_classification,take_smaller)
remove_genre = data.drop('분류기호',axis)
remove_genre = data.drop('분류기호',axis=1)
remove_genre['장르'] = genre_classification
pre_data = remove_genre
pre_data=pre_data[['대출자의키', '서명', '장르', '저작자','발행자']]
temp = data.groupby('대출자의키').count()
Tramsaction = pre_data.groupby(['대출자의키', '장르'])['장르'].count().reset_index(name='Count')
Transaction = pre_data.groupby(['대출자의키', '장르'])['장르'].count().reset_index(name='Count')
basket = Transaction.pivot_table(index='Transaction', columns='장르', values='Count', aggfunc='sum').fillna(0)
basket = Transaction.pivot_table(index='대출자의키', columns='장르', values='Count', aggfunc='sum').fillna(0)
def genre_classification(x):
    if x == 0:
        x = '사회'
        return x
    elif x >= 1 and x <= 3.99:
        x = '교양' 
        return x
    elif x >= 4 and x < 7:
        x = '수험서' 
        return x
    elif x >= 7 and x <= 30:
        x = '문헌정보' 
        return x
    elif x >= 31 and x <= 41:
        x = '교양' 
        return x
    elif x >= 80 and x <= 99:
        x = '만화' 
        return x
    elif x >= 100 and x <= 167:
        x = '철학' 
        return x
    elif x >= 180 and x < 200:
        x = '심리학' 
        return x
    elif x >= 200 and x < 300:
        x = '종교' 
        return x
    elif x >= 300 and x <= 310:
        x = '통계학' 
        return x
    elif x >= 311 and x <= 325.19:
        x = '경제,경영' 
        return x
    elif x >= 325.2 and x <= 326.79:
        x = '자기계발' 
        return x
    elif x >= 327 and x < 330:
        x = '경제,경영' 
        return x
    elif x >= 330 and x <= 390:
        x = '사회과학' 
        return x
    elif x >= 400 and x <= 499:
        x = '순수과학' 
        return x
    elif x >= 500 and x <= 599:
        x = '기술과학' 
        return x
    elif x >= 600 and x < 700:
        x = '예술' 
        return x
    elif x >= 700 and x <= 799:
        x = '언어'
        return x
    elif x >= 800 and x <= 899:
        x = '문학' 
        return x
    elif x >= 900 and x <= 999:
        x = '역사' 
        return x
    elif x >= 1000 and x <= 69999:
        x = '정부행정문서' 
        return x
    else:
        return x
genre_classification = data["분류기호"].apply(genre_classification) # 분류기호 숫자에 따라 장르에 맞게 변경

remove_genre = data.drop('분류기호',axis=1) # 열 이름 분류기호말고 장르로 바꾸는 과정
remove_genre['장르'] = genre_classification

pre_data = remove_genre # 열 순서가 마음에 안들어서 변경

pre_data=pre_data[['대출자의키', '서명', '장르', '저작자','발행자']]


Transaction = pre_data.groupby(['대출자의키', '장르'])['장르'].count().reset_index(name='Count')


basket = Transaction.pivot_table(index='대출자의키', columns='장르', values='Count', aggfunc='sum').fillna(0)

def encoding(x):
    if x<=0:
        return 0
    if x>=1:
        return 1

# applying the function to the dataset

basket = basket.applymap(encoding)
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
frequent_itemsets
rules.sort_values('confidence', ascending = False, inplace = True)
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.sort_values('confidence', ascending = False, inplace = True)
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
sorted_rules = rules.sort_values('confidence', ascending = False, inplace = True)
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
frequent_itemsets = apriori(basket, min_support=0.1, use_colnames=True)

frequent_itemsets

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)

## ---(Thu May 26 23:42:09 2022)---

"""
Created on Wed May 25 19:00:19 2022

@author: 이종웅
"""

import pandas as pd
import numpy as np

raw_data = pd.read_csv('C:/Users/이종웅/OneDrive - 서울과학기술대학교/서울과학기술대학교/서울과학기술대학교/데이터마이닝_김경옥/기말 과제/도서관 데이터/충청북도_행정간행물홈페이지_도서대여목록_20210601.csv')

# 저작자, 분류기호, 발행자, 서명, 대출자의키 뽑아내기
data = raw_data[['저작자', '분류기호', '발행자', '서명', '대출자의키']]

temp = data.groupby('대출자의키').count()
idx_num_1 = temp[temp['저작자']==1].index # 1권만 빌린 사람 제거
temp2 = temp.drop(idx_num_1) # 그 행은 삭제

user_index = temp2.reset_index() # 복잡한 대출자의 키를 간단한 인덱스로 변경
#pre_data = temp3.drop('대출자의키',axis=1)

def genre_classification(x):
    if x == 0:
        x = '사회'
        return x
    elif x >= 1 and x <= 3.99:
        x = '교양' 
        return x
    elif x >= 4 and x < 7:
        x = '수험서' 
        return x
    elif x >= 7 and x <= 30:
        x = '문헌정보' 
        return x
    elif x >= 31 and x <= 41:
        x = '교양' 
        return x
    elif x >= 80 and x <= 99:
        x = '만화' 
        return x
    elif x >= 100 and x <= 167:
        x = '철학' 
        return x
    elif x >= 180 and x < 200:
        x = '심리학' 
        return x
    elif x >= 200 and x < 300:
        x = '종교' 
        return x
    elif x >= 300 and x <= 310:
        x = '통계학' 
        return x
    elif x >= 311 and x <= 325.19:
        x = '경제,경영' 
        return x
    elif x >= 325.2 and x <= 326.79:
        x = '자기계발' 
        return x
    elif x >= 327 and x < 330:
        x = '경제,경영' 
        return x
    elif x >= 330 and x <= 390:
        x = '사회과학' 
        return x
    elif x >= 400 and x <= 499:
        x = '순수과학' 
        return x
    elif x >= 500 and x <= 599:
        x = '기술과학' 
        return x
    elif x >= 600 and x < 700:
        x = '예술' 
        return x
    elif x >= 700 and x <= 799:
        x = '언어'
        return x
    elif x >= 800 and x <= 899:
        x = '문학' 
        return x
    elif x >= 900 and x <= 999:
        x = '역사' 
        return x
    elif x >= 1000 and x <= 69999:
        x = '정부행정문서' 
        return x
    else:
        return x

genre_classification = data["분류기호"].apply(genre_classification) # 분류기호 숫자에 따라 장르에 맞게 변경

remove_genre = data.drop('분류기호',axis=1) # 열 이름 분류기호말고 장르로 바꾸는 과정
remove_genre['장르'] = genre_classification

pre_data = remove_genre # 열 순서가 마음에 안들어서 변경

pre_data=pre_data[['대출자의키', '서명', '장르', '저작자','발행자']]

# 트랜잭션 데이터 생성
Transaction = pre_data.groupby(['대출자의키', '장르'])['장르'].count().reset_index(name='Count')


basket = Transaction.pivot_table(index='대출자의키', columns='장르', values='Count', aggfunc='sum').fillna(0)


# 0, 1 로 변환해주는 함수
def encoding(x):
    if x<=0:
        return 0
    if x>=1:
        return 1

# applying the function to the dataset

basket = basket.applymap(encoding)


# 연관분석 시작
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(basket, min_support=0.1, use_colnames=True) #최소 지지도 0.1로 설

frequent_itemsets

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
raw_data
print(raw_data)
basket = Transaction.pivot_table(index='대출자의키', columns='장르', values='Count', aggfunc='sum').fillna(0)
basket2 = basket.applymap(encoding)
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(basket, min_support=0.1, use_colnames=True) #최소 지지도 0.1로 설

frequent_itemsets

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
Transcation_index []
Transcation_index[]
Transcation_index = []
Transcation_index(range(7153))
Transcation_index = list(range(7153))
pre_data['Transcation'] = Transcation_index
Transcation_index = list(range(7152))
pre_data['Transcation'] = Transcation_index
pre_data['Transcation_num'] = Transcation_index
pre_data=pre_data[['Transcation_num', '서명', '장르', '저작자','발행자']]
Transaction = pre_data.groupby(['Transcation_num', '장르'])['장르'].count().reset_index(name='Count')
basket = Transaction.pivot_table(index='Transcation_num', columns='장르', values='Count', aggfunc='sum').fillna(0)

## ---(Fri May 27 01:15:45 2022)---

"""
Created on Wed May 25 19:00:19 2022

@author: 이종웅
"""

import pandas as pd
import numpy as np

raw_data = pd.read_csv('C:/Users/이종웅/OneDrive - 서울과학기술대학교/서울과학기술대학교/서울과학기술대학교/데이터마이닝_김경옥/기말 과제/도서관 데이터/충청북도_행정간행물홈페이지_도서대여목록_20210601.csv')

# 저작자, 분류기호, 발행자, 서명, 대출자의키 뽑아내기
data = raw_data[['저작자', '분류기호', '발행자', '서명', '대출자의키']]

temp = data.groupby('대출자의키').count()
idx_num_1 = temp[temp['저작자']==1].index # 1권만 빌린 사람 제거
temp2 = temp.drop(idx_num_1) # 그 행은 삭제

user_index = temp2.reset_index() # 복잡한 대출자의 키를 간단한 인덱스로 변경
#pre_data = temp3.drop('대출자의키',axis=1)

def genre_classification(x):
    if x == 0:
        x = '사회'
        return x
    elif x >= 1 and x <= 3.99:
        x = '교양' 
        return x
    elif x >= 4 and x < 7:
        x = '수험서' 
        return x
    elif x >= 7 and x <= 30:
        x = '문헌정보' 
        return x
    elif x >= 31 and x <= 41:
        x = '교양' 
        return x
    elif x >= 80 and x <= 99:
        x = '만화' 
        return x
    elif x >= 100 and x <= 167:
        x = '철학' 
        return x
    elif x >= 180 and x < 200:
        x = '심리학' 
        return x
    elif x >= 200 and x < 300:
        x = '종교' 
        return x
    elif x >= 300 and x <= 310:
        x = '통계학' 
        return x
    elif x >= 311 and x <= 325.19:
        x = '경제,경영' 
        return x
    elif x >= 325.2 and x <= 326.79:
        x = '자기계발' 
        return x
    elif x >= 327 and x < 330:
        x = '경제,경영' 
        return x
    elif x >= 330 and x <= 390:
        x = '사회과학' 
        return x
    elif x >= 400 and x <= 499:
        x = '순수과학' 
        return x
    elif x >= 500 and x <= 599:
        x = '기술과학' 
        return x
    elif x >= 600 and x < 700:
        x = '예술' 
        return x
    elif x >= 700 and x <= 799:
        x = '언어'
        return x
    elif x >= 800 and x <= 899:
        x = '문학' 
        return x
    elif x >= 900 and x <= 999:
        x = '역사' 
        return x
    elif x >= 1000 and x <= 69999:
        x = '정부행정문서' 
        return x
    else:
        return x

genre_classification = data["분류기호"].apply(genre_classification) # 분류기호 숫자에 따라 장르에 맞게 변경

remove_genre = data.drop('분류기호',axis=1) # 열 이름 분류기호말고 장르로 바꾸는 과정
remove_genre['장르'] = genre_classification

pre_data = remove_genre # 열 순서가 마음에 안들어서 변경

pre_data=pre_data[['대출자의키', '서명', '장르', '저작자','발행자']]

# 트랜잭션 데이터 생성
Transaction = pre_data.groupby(['대출자의키', '장르'])['장르'].count().reset_index(name='Count')


basket = Transaction.pivot_table(index='대출자의키', columns='장르', values='Count', aggfunc='sum').fillna(0)


# 0, 1 로 변환해주는 함수
def encoding(x):
    if x<=0:
        return 0
    if x>=1:
        return 1

# applying the function to the dataset

basket = basket.applymap(encoding)


# 연관분석 시작
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(basket, min_support=0.1, use_colnames=True) #최소 지지도 0.1로 설

frequent_itemsets

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
frequent_itemsets
frequent_itemsets.sort_values('support', ascending=0)
rules.sort_values('support','confidence', ascending=0)
rules.sort_values('support', ascending=0)
sorted_rules_by_support = rules.sort_values('support', ascending=0)
sorted_rules_by_support.sort_values('confidence', ascending=0)
sorted_rules_by_support_confidence = sorted_rules_by_support.sort_values('confidence', ascending=0)
sorted_rules_by_support_confidence = rules.sort_values(by=["support", "confidence"], ascending=0) # 지지도와 신뢰도 둘 다 동시에 높은 순으로.
sorted_rules_by_support_confidence = rules.sort_values(by=["support", "confidence"], ascending=[1,1]) # 지지도와 신뢰도 둘 다 동시에 높은 순으로.
sorted_rules_by_support_confidence = rules.sort_values(by=["support", "confidence"], ascending=[0,0]) # 지지도와 신뢰도 둘 다 동시에 높은 순으로.

## ---(Sat May 28 01:19:18 2022)---

"""
Created on Wed May 25 19:00:19 2022

@author: 이종웅
"""

import pandas as pd
import numpy as np

raw_data = pd.read_csv('C:/Users/이종웅/OneDrive - 서울과학기술대학교/서울과학기술대학교/서울과학기술대학교/데이터마이닝_김경옥/기말 과제/도서관 데이터/충청북도_행정간행물홈페이지_도서대여목록_20210601.csv')

# 저작자, 분류기호, 발행자, 서명, 대출자의키 뽑아내기
data = raw_data[['저작자', '분류기호', '발행자', '서명', '대출자의키']]

temp = data.groupby('대출자의키').count()
idx_num_1 = temp[temp['저작자']==1].index # 1권만 빌린 사람 제거
temp2 = temp.drop(idx_num_1) # 그 행은 삭제

user_index = temp2.reset_index() # 복잡한 대출자의 키를 간단한 인덱스로 변경
#pre_data = temp3.drop('대출자의키',axis=1)

def genre_classification(x):
    if x == 0:
        x = '사회'
        return x
    elif x >= 1 and x <= 3.99:
        x = '교양' 
        return x
    elif x >= 4 and x < 7:
        x = '수험서' 
        return x
    elif x >= 7 and x <= 30:
        x = '문헌정보' 
        return x
    elif x >= 31 and x <= 41:
        x = '교양' 
        return x
    elif x >= 80 and x <= 99:
        x = '만화' 
        return x
    elif x >= 100 and x <= 167:
        x = '철학' 
        return x
    elif x >= 180 and x < 200:
        x = '심리학' 
        return x
    elif x >= 200 and x < 300:
        x = '종교' 
        return x
    elif x >= 300 and x <= 310:
        x = '통계학' 
        return x
    elif x >= 311 and x <= 325.19:
        x = '경제,경영' 
        return x
    elif x >= 325.2 and x <= 326.79:
        x = '자기계발' 
        return x
    elif x >= 327 and x < 330:
        x = '경제,경영' 
        return x
    elif x >= 330 and x <= 390:
        x = '사회과학' 
        return x
    elif x >= 400 and x <= 499:
        x = '순수과학' 
        return x
    elif x >= 500 and x <= 599:
        x = '기술과학' 
        return x
    elif x >= 600 and x < 700:
        x = '예술' 
        return x
    elif x >= 700 and x <= 799:
        x = '언어'
        return x
    elif x >= 800 and x <= 899:
        x = '문학' 
        return x
    elif x >= 900 and x <= 999:
        x = '역사' 
        return x
    elif x >= 1000 and x <= 69999:
        x = '정부행정문서' 
        return x
    else:
        return x

genre_classification = data["분류기호"].apply(genre_classification) # 분류기호 숫자에 따라 장르에 맞게 변경

remove_genre = data.drop('분류기호',axis=1) # 열 이름 분류기호말고 장르로 바꾸는 과정
remove_genre['장르'] = genre_classification

pre_data = remove_genre # 열 순서가 마음에 안들어서 변경

pre_data=pre_data[['대출자의키', '서명', '장르', '저작자','발행자']]

# 트랜잭션 데이터 생성
Transaction = pre_data.groupby(['대출자의키', '장르'])['장르'].count().reset_index(name='Count')


basket = Transaction.pivot_table(index='대출자의키', columns='장르', values='Count', aggfunc='sum').fillna(0)


# 0, 1 로 변환해주는 함수
def encoding(x):
    if x<=0:
        return 0
    if x>=1:
        return 1

# applying the function to the dataset

basket = basket.applymap(encoding)


# 연관분석 시작
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(basket, min_support=0.1, use_colnames=True) #최소 지지도 0.1로 설

frequent_itemsets.sort_values('support', ascending=0)

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1) #min_threshold = 1 은 lift의 최솟값이 1인 것이다.


sorted_rules_by_support = rules.sort_values('support', ascending=0) # 지지도만 높은 순으로
sorted_rules_by_support_confidence = rules.sort_values(by=["support", "confidence"], ascending=[0,0]) # 지지도와 신뢰도 둘 다 동시에 높은 순으로.
frequent_itemsets = apriori(basket, min_support=0.5, use_colnames=True) #최소 지지도 0.1로 설

frequent_itemsets.sort_values('support', ascending=0)

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1) #min_threshold = 1 은 lift의 최솟값이 1인 것이다.
sorted_rules_by_support = rules.sort_values('support', ascending=0) # 지지도만 높은 순으로
frequent_itemsets = apriori(basket, min_support=0.3, use_colnames=True) #최소 지지도 0.1로 설
frequent_itemsets = apriori(basket, min_support=0.2, use_colnames=True) #최소 지지도 0.1로 설
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1) #min_threshold = 1 은 lift의 최솟값이 1인 것이다.
sorted_rules_by_support = rules.sort_values('support', ascending=0) # 지지도만 높은 순으로
frequent_itemsets.sort_values('support', ascending=0)
sorted_rules_by_support_confidence = rules.sort_values(by=["support", "confidence"], ascending=[0,0]) # 지지도와 신뢰도 둘 다 동시에 높은 순으로.
frequent_itemsets.sort_values('support', ascending=0)
frequent_itemsets = apriori(basket, min_support=0.5, use_colnames=True) #최소 지지도 0.1로 설

## ---(Mon Aug 22 13:06:53 2022)---
runfile('C:/Users/이종웅/untitled0.py', wdir='C:/Users/이종웅')

## ---(Thu Aug 25 17:42:21 2022)---
runfile('C:/Users/이종웅/.spyder-py3/temp.py', wdir='C:/Users/이종웅/.spyder-py3')
runcell(0, 'C:/Users/이종웅/.spyder-py3/temp.py')
runfile('C:/Users/이종웅/.spyder-py3/temp.py', wdir='C:/Users/이종웅/.spyder-py3')
runcell(0, 'C:/Users/이종웅/.spyder-py3/temp.py')
runfile('C:/Users/이종웅/.spyder-py3/temp.py', wdir='C:/Users/이종웅/.spyder-py3')
runcell(0, 'C:/Users/이종웅/.spyder-py3/temp.py')
runfile('C:/Users/이종웅/.spyder-py3/history.py', wdir='C:/Users/이종웅/.spyder-py3')
runfile('C:/Users/이종웅/.spyder-py3/Self practice week 3.py', wdir='C:/Users/이종웅/.spyder-py3')
runfile('C:/Users/이종웅/.spyder-py3/untitled0.py', wdir='C:/Users/이종웅/.spyder-py3')

## ---(Fri Aug 26 11:36:22 2022)---
runfile('C:/Users/이종웅/.spyder-py3/독립표본 t-test.py', wdir='C:/Users/이종웅/.spyder-py3')
runcell(0, 'C:/Users/이종웅/.spyder-py3/독립표본 t-test.py')
debugfile('C:/Users/이종웅/.spyder-py3/독립표본 t-test.py', wdir='C:/Users/이종웅/.spyder-py3')

## ---(Fri Aug 26 13:08:59 2022)---
runfile('C:/Users/이종웅/.spyder-py3/독립표본 t-test.py', wdir='C:/Users/이종웅/.spyder-py3')
runcell(0, 'C:/Users/이종웅/.spyder-py3/독립표본 t-test.py')
runfile('C:/Users/이종웅/.spyder-py3/독립표본 t-test.py', wdir='C:/Users/이종웅/.spyder-py3')
runcell(0, 'C:/Users/이종웅/.spyder-py3/독립표본 t-test.py')
runcell(0, 'C:/Users/이종웅/.spyder-py3/대응표본 t-test.py')
runfile('C:/Users/이종웅/.spyder-py3/대응표본 t-test.py', wdir='C:/Users/이종웅/.spyder-py3')
runcell(0, 'C:/Users/이종웅/.spyder-py3/대응표본 t-test.py')
runcell(0, 'C:/Users/이종웅/.spyder-py3/일원 분산 분석(one-way ANOVA).py')

## ---(Mon Aug 29 19:22:42 2022)---
runcell(0, 'C:/Users/이종웅/.spyder-py3/일원 분산 분석(one-way ANOVA).py')

## ---(Wed Aug 31 16:48:33 2022)---
runcell(0, 'C:/Users/이종웅/.spyder-py3/일원 분산 분석(one-way ANOVA).py')

## ---(Wed Aug 31 18:37:38 2022)---
runcell(0, 'C:/Users/이종웅/.spyder-py3/일원 분산 분석(one-way ANOVA).py')
runcell(0, 'C:/Users/이종웅/.spyder-py3/선형회귀.py')

## ---(Fri Sep  2 18:46:43 2022)---
runcell(0, 'C:/Users/이종웅/.spyder-py3/선형회귀.py')

## ---(Sat Sep  3 07:13:00 2022)---
runcell(0, 'C:/Users/이종웅/.spyder-py3/선형회귀.py')

## ---(Wed Sep  7 05:40:50 2022)---
runfile('C:/Users/이종웅/.spyder-py3/선형회귀.py', wdir='C:/Users/이종웅/.spyder-py3')
runcell(0, 'C:/Users/이종웅/.spyder-py3/선형회귀.py')

## ---(Sun Sep 18 22:20:55 2022)---
runcell(0, 'C:/Users/이종웅/.spyder-py3/untitled0.py')

## ---(Thu Sep 22 20:59:51 2022)---
runcell(0, 'C:/Users/이종웅/.spyder-py3/직접 만든 표본분산 계산기.py')
runfile('C:/Users/이종웅/.spyder-py3/직접 만든 표본분산 계산기.py', wdir='C:/Users/이종웅/.spyder-py3')
runcell(0, 'C:/Users/이종웅/.spyder-py3/직접 만든 표본분산 계산기.py')

## ---(Fri Sep 23 12:06:48 2022)---
runcell(0, 'C:/Users/이종웅/.spyder-py3/untitled0.py')
runcell(0, 'C:/Users/이종웅/.spyder-py3/untitled1.py')
runcell(0, 'C:/Users/이종웅/.spyder-py3/untitled0.py')
runcell(0, 'C:/Users/이종웅/.spyder-py3/직접 만든 표본분산 계산기.py')
runcell(0, 'C:/Users/이종웅/.spyder-py3/untitled0.py')

## ---(Fri Sep 23 17:47:49 2022)---
runcell(0, 'C:/Users/이종웅/.spyder-py3/통활과제흔적.py')

## ---(Wed Sep 28 09:19:48 2022)---
runcell(0, 'C:/Users/이종웅/.spyder-py3/직접 만든 표본분산 계산기.py')
runcell(0, 'C:/Users/이종웅/.spyder-py3/통활과제흔적.py')
runcell(0, 'C:/Users/이종웅/.spyder-py3/untitled0.py')
runcell(0, 'C:/Users/이종웅/.spyder-py3/상관분석 3차과제 사이파이를 이용한 상관분석검정.py')

## ---(Sat Oct  8 22:29:37 2022)---
runcell(0, 'C:/Users/이종웅/.spyder-py3/단순선형회귀분석 통계학 교재 순서대로 코딩해보기.py')