# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 08:32:20 2019

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

# get a subset of the data for now, for speed
k = 1000
data = data.sample(k, random_state = 1)

start = datetime.now()
# plot data
x0 = 475826.427137
y0 = 6.923281e+06
#x0 = 475800.6
#y0 = 6924485.4

r0 = np.array([np.mean(data.x), np.mean(data.y)])
data['x'] = r0[0] - data.x # shift so that main shock position is (0,0)
data['y'] = r0[1] - data.y
data['Distance_from_origin'] = (data.x**2 + data.y**2)**0.5
eq.plot_catalog(data, 1, np.array([0,0]), color = 'Generation')

r, densities = eq.plot_ED(data, k = 20,  plot = False) # get distance, density and plot

# perform particle swarm optimisation in parameter space on log likelihood
rho0 = np.mean(densities[0:6])
rmax = (r.max())
rmin = (r.min())
n_edges = 10 
bin_edges = np.linspace(np.log10(rmin), np.log10(rmax), n_edges) #np.array([r[i] for i in range(0, len(r), q)])
bin_edges = 10**bin_edges
#bin_edges = np.linspace(rmin, rmax, n_edges) #np.array([r[i] for i in range(0, len(r), q)])
const = (rmax, rmin, r, rho0, bin_edges, n_edges)

lb = [1, 1]
ub = [800, 6]

# do particle swarm opti.
theta0, llk0 = pso(eq.LLK_rho, lb, ub, args = const, maxiter = 100, swarmsize = 200)

# plots
f, ax = plt.subplots(1, figsize = (7,7))

ax.plot(r, densities, 'o')

rplot = np.linspace((rmin),(rmax),500)
#ax.plot(rplot, (eq.rho(rplot, rho0, 290.4, 5.2)),'-',color='r')
ax.plot(rplot, (eq.rho(rplot, rho0, theta0[0], theta0[1])),'-',color='b')
ax.set_xscale('log')
ax.set_yscale('log')

print(datetime.now().timestamp() - start.timestamp())