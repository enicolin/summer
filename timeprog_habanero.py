# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 12:01:19 2019

@author: enic156
"""

import numpy as np
import pandas as pd
import eq_functions as eq
from datetime import datetime
from pyswarm import pso
import matplotlib.pyplot as plt

# read in data and convert to pandas catalog format
data_raw = pd.read_csv('habanero_data.csv')

data = data_raw.drop(['z','elapsed time (s)'], axis = 1)
data = data.rename(index = str, columns = {'elapsed time (s) corrected': 'Time'})
nrows = data.shape[0]
data['n_avg'] = pd.Series(['-'] * nrows, index = data.index)
data['Events'] = pd.Series(['-'] * nrows, index = data.index)
data['Magnitude'] = pd.Series([2] * nrows, index = data.index)
data['Generation'] = pd.Series([0] * nrows, index = data.index)
data['Distance'] = pd.Series(['-'] * nrows, index = data.index)

cols = ['n_avg','Events','Magnitude','Generation','x','y','Distance','Time','Distance_from_origin']
data = data.reindex(columns = cols)


start = datetime.now()
# plot data
#x0 = 475513.1
#y0 = 6922782.2
x0 = np.mean(data.x) 
y0 = np.mean(data.y) 

data['x'] = x0 - data.x # shift so that main shock position is (0,0)
data['y'] = y0 - data.y
data['Distance_from_origin'] = (data.x**2 + data.y**2)**0.5
data = data[data.Distance_from_origin < 10**3.2]

for i in range(322,4001):
    eq.plot_catalog(data[:i], 1, np.array([0,0]), color = 'Generation', savepath = 'habanero_frames/{}.png'.format(i), saveplot = True)

print(datetime.now() - start)