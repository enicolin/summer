# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 12:04:32 2019

@author: enic156
"""
import eq_functions as eq
import pandas as pd
from datetime import datetime
import numpy as np
import random
from pyswarm import pso
import matplotlib.pyplot as plt

np.random.seed(1756)
random.seed(1756)

# read in fenton hill data and store event objects in list
f = open("FentonHillExpt2032-MHF-goodlocs.txt",'r')
flines = f.readlines()
flines = flines[0::2]

events = [eq.Event(float(line.split()[11]), line.split()[1][0:2], float(line.split()[6]), float(line.split()[5]), '-', 0, '-') for line in flines]

f.close()

start = datetime.now()

# reduce events to a random sample of k elements
k = 650
events = random.sample(events, k)

# format events in the pd dataframe format defined by generate_catalog etc. 
catalog = pd.DataFrame({'Magnitude': [2.3] * len(events),
                                   'Events':'-',
                                   'n_avg':'-',
                                   'Time':[event.time for event in events],
                                   'Distance':['-'] * len(events),
                                   'x':[event.x for event in events],
                                   'y':[event.y for event in events],
                                   'Generation':[0] * len(events),
                                   'Distance_from_origin': [event.distance_from_origin for event in events]})
cols = ['n_avg','Events','Magnitude','Generation','x','y','Distance','Time','Distance_from_origin']
catalog = catalog.reindex(columns = cols)
catalog = catalog[catalog.Time == '83']

r0 = np.array([np.mean([event.x for event in events]), np.mean([event.y for event in events])])#np.array([3557.418383, -324.384367])
catalog['x'] = r0[0]+58 - catalog.x # shift so that main shock position is (0,0)
catalog['y'] = r0[1]-130 - catalog.y
catalog['Distance_from_origin'] = (catalog.x**2 + catalog.y**2)**0.5
catalog = catalog[catalog.Distance_from_origin <= 10**2.6]
eq.plot_catalog(catalog, 1, np.array([0,0]), color = 'Generation')

r, densities = eq.plot_ED(catalog, k = 40,  plot = False) # get distance, density

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
ub = [1000, 6]

# do particle swarm opti.
theta0, llk0 = pso(eq.LLK_rho, lb, ub, args = const, maxiter = 100, swarmsize = 150)

# plots
f, ax = plt.subplots(1, figsize = (7,7))

ax.plot(r, densities, 'o')

#theta0 = np.array([140, 2.4])
rplot = np.linspace((rmin),(rmax),500)
#ax.plot(rplot, (eq.rho(rplot, rho0, 238.6, 2.6)),'-',color='r') # raft river
#ax.plot(rplot, (eq.rho(rplot, rho0, 142, 2.8)),'-',color='r') # newberry
ax.plot(rplot, (eq.rho(rplot, rho0, theta0[0], theta0[1], plot = True)),'-',color='r')
for be in bin_edges:
    ax.axvline(be,color='k',linestyle=':')
ax.set_xscale('log')
ax.set_yscale('log')
#print('theta0 = {}, llk = {}'.format(theta0,llk0))
    
print(datetime.now().timestamp() - start.timestamp())

#f, ax = plt.subplots(1, figsize = (7,7))
#n = 125
#rc = np.linspace(50, 400, n)
#gmma = np.linspace(1, 6, n)
#X, Y = np.meshgrid(rc, gmma)
#Z = np.zeros(np.shape(X))
#for i in range(n):
#    for j in range(n):
#        Z[i][j] = eq.LLK_rho([X[i][j],Y[i][j]], rmax, rmin, r, rho0, bin_edges, n_edges)
#cs = plt.contourf(X,Y,Z,80,colormap = 'plasma')
#f.colorbar(cs, ax=ax)
#plt.show()