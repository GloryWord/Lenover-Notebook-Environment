# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 13:10:00 2022

@author: 이종웅
"""

import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([2.4, 1.2 ,3.5, 2.1, 4.7])
y_data = np.zeros(5)

mu = 0 # 평균값
sigma = 1 # 표준편차

def pdf(x, mu, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2 / (2*sigma**2))

mu = np.average(x_data)
sigma = np.std(x_data)

def log_likelihood(p): # 대수우도 구하는 함수
    return np.sum(np.log(p)) 

x_sigma = np.linspace(0.5, 8) # 가로축에 사용하는 표준편차, 0.5부터 8까지 50 분등
y_loglike = [] # 세로축에 사용하는 대수우도
for s in x_sigma: # x_sigma에서 x는 가로축에 대한 것이고 s는 그냥 0부터 x_sigma 배열 크기만큼 반복 하는것. 
    # 좌표에 x를 50칸 찍기 위해 반복하는 것이다. 그 간격은 linspace로 나눔.
    log_like = log_likelihood(pdf(x_data, mu, s))
    print("{}번째 log_likelihood 인수의 값은 {}, 그래서 log_like에 들어가는 값은 {} \n".format(s, pdf(x_data, mu, s), log_like))
    y_loglike.append(log_like)
    
pdf_list = pdf(x_data, mu, s)
log_pdf_list = np.log(pdf(x_data, mu, s))
sum_of_log_pdf_list = np.sum(log_pdf_list)

