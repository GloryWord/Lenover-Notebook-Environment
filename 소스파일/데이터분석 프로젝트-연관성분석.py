# -*- coding: utf-8 -*-
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
    


