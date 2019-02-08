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


start = datetime.now()
x0 = np.mean(data.x) - 50
y0 = np.mean(data.y) -250

data['x'] = x0 - data.x # shift so that main shock position is (0,0)
data['y'] = y0 - data.y
data['Distance_from_origin'] = (data.x**2 + data.y**2)**0.5
data = data.dropna()
data = data.sample(frac = 0.3, replace = False)

eq.plot_catalog(data, 1, np.array([0,0]), color = 'Generation')

N = len(data)
k = 40 #int(N**0.5)
r, densities = eq.plot_ED(data, k = k,  plot = False) # get distance, density
df_dens = pd.DataFrame({'distance':r, 'density':densities})
df_dens = df_dens[(df_dens.distance > 10**1.4) & (df_dens.distance <= 10**3)]
r = np.array(df_dens.distance)
densities = np.array(df_dens.density) 

#==============================================================================
# rho0 = np.mean(densities[0:100])
# rmax = (r.max())
# rmin = (r.min())
# n_edges = 30
# #bin_edges = np.linspace(np.log10(rmin), np.log10(rmax), n_edges) #np.array([r[i] for i in range(0, len(r), q)])
# #bin_edges = 10**bin_edges
# bin_edges = np.linspace(rmin, rmax, n_edges) #np.array([r[i] for i in range(0, len(r), q)])
# const = (rmax, rmin, r, bin_edges, n_edges, rho0)
# 
# lb = [50, 1]#, 1e-4]
# ub = [900, 6]#, 1]
# 
# # do particle swarm opti.
# #theta0, obj = pso(eq.LLK_rho, lb, ub, args = const, maxiter = 100, swarmsize = 100)
# 
# # plots
# f, ax = plt.subplots(1, figsize = (7,4))
# 
# theta1 = np.array([634.5, 4.9])
# ax.plot(r, densities, 'o') #, alpha = 0.6, color = 'k')
# rplot = np.linspace((rmin),(rmax),500)
# #ax.plot(rplot, (eq.rho(rplot, rho0, theta0[0], theta0[1])),'-',color='r')
# #ax.plot(rplot, (eq.rho(rplot, rho0, theta1[0], theta1[1])),'-',color='b')
# for be in bin_edges:
#     ax.axvline(be,color='k',linestyle=':')
# ax.set_xscale('log')
# ax.set_yscale('log')
#==============================================================================

rho0 = np.mean(densities[0:100])
rmax = (r.max())
rmin = (r.min())
n_edges = 10
bin_edges = np.linspace(np.log10(rmin), np.log10(rmax), n_edges) #np.array([r[i] for i in range(0, len(r), q)])
bin_edges = 10**bin_edges
#bin_edges = np.linspace(rmin, rmax, n_edges) #np.array([r[i] for i in range(0, len(r), q)])
const = (r, densities, bin_edges)

lb = [1, 1, 1e-8]
ub = [1000, 6, 1]

# do particle swarm opti.
theta0, obj = pso(eq.robj, lb, ub, args = const, maxiter = 100, swarmsize = 150)

#    # plots
f, ax = plt.subplots(1, figsize = (7,4))

theta1 = np.array([212.8, 4.4, rho0])
ax.plot(r, densities, 'o') #, alpha = 0.6, color = 'k')
rplot = np.linspace((rmin),(rmax),500)

eq.robj((theta0[0], theta0[1], rho0), r, densities, bin_edges)
ax.plot(rplot, (eq.rho(rplot, theta0[2], theta0[0], theta0[1])),'-',color='r')
#ax.plot(rplot, (eq.rho(rplot, theta1[2], theta1[0], theta1[1])),'-',color='b')
#for be in bin_edges:
#    ax.axvline(be,color='k',linestyle=':')
ax.set_xscale('log')
ax.set_yscale('log')

print(datetime.now().timestamp() - start.timestamp())