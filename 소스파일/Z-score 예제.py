# -*- coding: utf-8 -*-
"""
Created on Sat May 14 18:02:48 2022

@author: 이종웅
"""

import numpy as np
from numpy import *
data = np.random.randint(30, size=(6, 5))
data

data_standadized_np = (data - mean(data,axis=0)) / std(data,axis=0)
data_standadized_np.mean()
data_standadized_np.std()
