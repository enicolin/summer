# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 08:32:20 2019

@author: enic156
"""
import numpy as np
import pandas as pd
import eq_functions as eq
from datetime import datetime

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

# get a subset of the data for now, for speed
k = 1000
data = data.sample(k, random_state = 1)

start = datetime.now()
# plot data
r0 = np.array([475826.427137,6.923281e+06]) # "initial shock" - mean (x,y) position of data
data['x'] = r0[0] - data.x # shift so that main shock position is (0,0)
data['y'] = r0[1] - data.y
data['Distance_from_origin'] = (data.x**2 + data.y**2)**0.5
eq.plot_catalog(data, 2, np.array([0,0]), color = 'Generation')

# plot density plot
eq.plot_ED(data)

print(datetime.now().timestamp() - start.timestamp())