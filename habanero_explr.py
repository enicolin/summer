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
from scipy import optimize
import random

random.seed(4)
np.random.seed(4)

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
x0 = np.mean(data.x) - 20
y0 = np.mean(data.y) -150

data['x'] = x0 - data.x # shift so that main shock position is (0,0)
data['y'] = y0 - data.y
data['Distance_from_origin'] = (data.x**2 + data.y**2)**0.5
data['Distance_from_origin'] /= 1000 # set in km
data = data.dropna()
data = data.sample(frac = 0.03, replace = False)

eq.plot_catalog(data, 1, np.array([0,0]), color = 'Generation')

N = len(data)
k = 40 #int(N**0.5)
r, densities = eq.plot_ED(data, k = k,  plot = False) # get distance, density
df_dens = pd.DataFrame({'distance':r, 'density':densities})
df_dens = df_dens[(df_dens.distance > 10**1.4) & (df_dens.distance <= 10**3)]
r = np.array(df_dens.distance)
densities = np.array(df_dens.density) 

#==============================================================================
## perform particle swarm optimisation (MLE)
#rho0 = np.mean(densities[0:5])
#rmax = (r.max())
#rmin = (r.min())
#n_edges = 15
##bin_edges = np.linspace(np.log10(rmin), np.log10(rmax), n_edges) #np.array([r[i] for i in range(0, len(r), q)])
##bin_edges = 10**bin_edges
#bin_edges = np.linspace(rmin, rmax, n_edges) #np.array([r[i] for i in range(0, len(r), q)])
#const = (rmax, rmin, r, bin_edges, n_edges, rho0)
#
#lb = [50, 1]
#ub = [900, 6]
#bounds = [(low, high) for low, high in zip(lb,ub)] # basinhopping bounds
#
## do particle swarm opti.
#theta0, obj = pso(eq.LLK_rho, lb, ub, args = const, maxiter = 100, swarmsize = 500)
##minimizer_kwargs = {"args":const, "bounds":bounds, "method":"L-BFGS-B"}
##annealed = optimize.basinhopping(eq.LLK_rho, theta_guess, minimizer_kwargs = minimizer_kwargs, niter = 500)
##theta0 = annealed.x
##theta1 = np.array([634.5, 4.9])
## plots
#f, ax = plt.subplots(1, figsize = (7,4))
#
#ax.plot(r, densities, 'o') 
##ax.set_ylim(ax.get_ylim())
#rplot = np.linspace((rmin),(rmax),500)
#ax.plot(rplot, (eq.rho(rplot, rho0, theta0[0], theta0[1])),'-',color='r')
#for be in bin_edges:
#    ax.axvline(be,color='k',linestyle=':')
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.set_title('Habanero')
#==============================================================================
## minimise weighted sum of squares
##rho0 = np.mean(densities[0:100])
#rmax = (r.max())
#rmin = (r.min())
#n_edges = 32
#bin_edges = np.linspace(np.log10(rmin), np.log10(rmax), n_edges) #np.array([r[i] for i in range(0, len(r), q)])
#bin_edges = 10**bin_edges
##bin_edges = np.linspace(rmin, rmax, n_edges) #np.array([r[i] for i in range(0, len(r), q)])
#const = (r, densities, bin_edges)
#
#lb = [1e-8, 1, 1e-19, 1e-10, 1e-9]
#ub = [1e2, 1e9, 1e2, 1e0, 1e3]
##lb = [1e-60,1e-60]
##ub = [1e60, 1e60]
#bounds = [(low, high) for low, high in zip(lb,ub)] # L-BFGS-B bounds
##alpha, T, k, nu, q  = theta
#
## do particle swarm opti.
#theta0, obj = pso(eq.robj_diff, lb, ub, args = const, maxiter = 500, swarmsize = 200, phip = 0.75, minfunc = 1e-12, minstep = 1e-12, phig = 0.8)
##theta_guess = np.array([  8.00000000e-11,   7.68417736e+08,   7.63685917e+01, 4.65534682e-01,   9.04412151e+02]) # 2D initial guess
##theta_guess = np.array([  8.00000000e-11,   19.68417736e+9]) # 2D initial guess
##theta_guess = np.array([  20.00000000e-14,   7.68417736e+08,   7.63685917e+01, 4.65534682e-01,   9.04412151e+02]) # 3D initial guess
##theta_guess = np.array([  7.50000000e-11,   19.68417736e+21]) # 3D initial guess
##minimizer_kwargs = {"args":const, "bounds":bounds, "method":"L-BFGS-B"}
##annealed = optimize.basinhopping(eq.robj_diff, theta_guess, minimizer_kwargs = minimizer_kwargs, niter = 1000, T =1e20)
##theta0 = annealed.x
#
## plots
#f, ax = plt.subplots(1, figsize = (7,4))
#
#ax.plot(r, densities, 'o') 
##ax.set_ylim(ax.get_ylim())
#rplot = np.linspace((rmin),(rmax),500)
#ax.plot(rplot, eq.p3D(rplot, theta0[0], theta0[1], theta0[2], theta0[3], theta0[4]),'-',color='r')
##ax.set_xscale('log')
##ax.set_yscale('log')
#==============================================================================

print(datetime.now().timestamp() - start.timestamp())