# -*- coding: utf-8 -*-
"""
Created on Fri May  6 21:07:02 2022

@author: 이종웅
"""

\
# 데이터에서 각 월들이 몇번씩 나왔는지 확인.
print(data['month'].value_counts())
print("\n")

# 1월 산불 발생 수 계산.
jan_non_event = data[(data['month']=='jan')&(data['area']==0)]
jan_event = 2 - len(jan_non_event)
print("1월 산불 발생 수 : ",jan_event, " 산불 비율 : ",round(jan_event/2,2) )

# 2월 산불 발생 수 계산.
feb_non_event = data[(data['month']=='feb')&(data['area']==0)]
feb_event = 20 - len(feb_non_event)
print("2월 산불 발생 수 : ",feb_event, " 산불 비율 : ",round(feb_event/20,2) )

# 3월 산불 발생 수 계산.
mar_non_event = data[(data['month']=='mar')&(data['area']==0)]
mar_event = 54 - len(mar_non_event)
print("3월 산불 발생 수 : ",mar_event, " 산불 비율 : ",round(mar_event/54,2) )

# 4월 산불 발생 수 계산.
apr_non_event = data[(data['month']=='apr')&(data['area']==0)]
apr_event = 9 - len(apr_non_event)
print("4월 산불 발생 수 : ",apr_event, " 산불 비율 : ",round(apr_event/9,2) )

# 5월 산불 발생 수 계산.
may_non_event = data[(data['month']=='may')&(data['area']==0)]
may_event = 2 - len(may_non_event)
print("5월 산불 발생 수 : ",may_event, " 산불 비율 : ",round(may_event/2,2) )

# 6월 산불 발생 수 계산.
jun_non_event = data[(data['month']=='jun')&(data['area']==0)]
jun_event = 17 - len(jun_non_event)
print("6월 산불 발생 수 : ",jun_event, " 산불 비율 : ",round(jun_event/17,2) )

# 7월 산불 발생 수 계산.
jul_non_event = data[(data['month']=='jul')&(data['area']==0)]
jul_event = 32 - len(jul_non_event)
print("7월 산불 발생 수 : ",jul_event, " 산불 비율 : ",round(jul_event/32,2) )

# 8월 산불 발생 수 계산.
aug_non_event = data[(data['month']=='aug')&(data['area']==0)]
aug_event = 184 - len(aug_non_event)
print("8월 산불 발생 수 : ",aug_event, " 산불 비율 : ",round(aug_event/184,2) )