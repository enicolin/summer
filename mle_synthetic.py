# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:12:17 2019

@author: enic156
"""

import eq_functions as eq
import pandas as pd
import numpy as np
import random as rnd
from datetime import datetime
from pyswarm import pso
import matplotlib.pyplot as plt

start = datetime.now()

np.random.seed(1756)
rnd.seed(1756)

# define parameters
#Nt = 1000
Tf = 100 # forecast period
b = 1.
c = 1.
cprime = 1.
p = 1.1
pprime = 1.8
Mc = 2.
smin = 0.5 # minimum seismicity allowable on an interval so that it doesn't get too small
M0 = 5.7 # magnitude of initial earthquake
A = 1.1 # parameter included in law for generating expected aftershocks given main shock magnitude M0
alpha = 1.4 # parameter included in law for generating expected aftershocks given main shock magnitude M0

t0 = 0 # time of main shock
r0 = np.array([0,0]) # x,y coord. of main shock
gen = 0 # initial generation

### generate catalog and save
#catalog_list = []
#eq.generate_catalog(t0, r0, catalog_list, gen, True,
#                    Tf,M0,A,alpha,b,c,cprime,p,pprime,Mc,smin)
#
#catalogs = pd.concat(catalog_list)
#catalogs.to_pickle('catalogs.pkl')


# read in catalog and plot
catalogs = pd.read_pickle('catalogs.pkl') # read in .pkl file storing dataframe generated above
catalogs = catalogs[catalogs.Magnitude > 0]
#catalogs = catalogs[catalogs.Distance_from_origin > 10]
# plot catalog
#eq.plot_catalog(catalogs_raw, M0, r0, color = 'Density')
#eq.plot_catalog(catalogs_raw, M0, r0, color = 'Time')

#eq.plot_catalog(catalog, 1, np.array([0,0]), color = 'Generation')
r, densities = eq.plot_ED(catalogs, k = 30,  plot = False) # get distance, density
r = r/r.min()

# perform particle swarm optimisation in parameter space on log likelihood
rho0 = np.mean(densities[0:6])
rmax = (r.max())
rmin = (r.min())
n_edges = 10
bin_edges = np.linspace(np.log10(rmin), np.log10(rmax), n_edges) #np.array([r[i] for i in range(0, len(r), q)])
bin_edges = 10**bin_edges
#bin_edges = np.linspace(rmin, rmax, n_edges) #np.array([r[i] for i in range(0, len(r), q)])
const = (rmax, rmin, r, rho0, bin_edges, n_edges)

lb = [1e-1, 1]
ub = [1000, 6]

# do particle swarm opti.
theta0, llk0 = pso(eq.LLK_rho, lb, ub, args = const, maxiter = 100, swarmsize = 150)

# plots
f, ax = plt.subplots(1, figsize = (7,7))

ax.plot(r, densities, 'o')

#theta0 = np.array([100, 2.4])
rplot = np.linspace((rmin),(rmax),500)
#ax.plot(rplot, (eq.rho(rplot, rho0, 238.6, 2.6)),'-',color='r') # raft river
#ax.plot(rplot, (eq.rho(rplot, rho0, 142, 2.8)),'-',color='r') # newberry
ax.plot(rplot, (eq.rho(rplot, rho0, theta0[0], theta0[1], plot = True)),'-',color='b')
for be in bin_edges:
    ax.axvline(be,color='k',linestyle=':')
ax.set_xscale('log')
ax.set_yscale('log')
print(theta0)
print(datetime.now() - start)