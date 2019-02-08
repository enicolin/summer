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

#==============================================================================
# read in fenton hill data and store event objects in list
f = open("FentonHillExpt2032-MHF-goodlocs.txt",'r')
flines = f.readlines()
flines = flines[0::2]

time_init = datetime(1983,12,6,hour=22,minute=4,second=36)
events = [eq.Event(float(line.split()[11]),\
                   (datetime(int('19'+line.split()[1][:2]), int(line.split()[1][2:]),int(line.split()[2][:2]),hour=int(line.split()[2][2:]),minute=int(line.split()[3][:2]),second=int(line.split()[3][2:]))-time_init).total_seconds()\
                   , float(line.split()[6]), float(line.split()[5]), '-', 0, '-') for line in flines]

f.close()

start = datetime.now()


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

r0 = np.array([np.mean([event.x for event in events]), np.mean([event.y for event in events])])#np.array([3557.418383, -324.384367])
catalog['x'] = r0[0]- catalog.x # shift so that main shock position is (0,0)
catalog['y'] = r0[1] - catalog.y
catalog['Distance_from_origin'] = (catalog.x**2 + catalog.y**2)**0.5
catalog = catalog.dropna()

# reduce events to a random sample of k elements
#catalog = catalog.sample(frac=0.45, replace=False)
#==============================================================================

eq.plot_catalog(catalog, 1, np.array([0,0]), color = 'Generation')

N = len(catalog)
k = 20 # int(N**0.5)

r, densities = eq.plot_ED(catalog, k = k,  plot = False) # get distance, density
df_dens = pd.DataFrame({'distance':r, 'density':densities})
#df_dens = df_dens[(df_dens.distance > 10**1.2) & (df_dens.distance < 10**2.6)]
r = df_dens.distance
densities = df_dens.density #* np.exp()

#rho0 = np.mean(densities[0:100])
#rmax = (r.max())
#rmin = (r.min())
#n_edges = 30
##bin_edges = np.linspace(np.log10(rmin), np.log10(rmax), n_edges) #np.array([r[i] for i in range(0, len(r), q)])
##bin_edges = 10**bin_edges
#bin_edges = np.linspace(rmin, rmax, n_edges) #np.array([r[i] for i in range(0, len(r), q)])
#const = (rmax, rmin, r, bin_edges, n_edges, rho0)
#
#lb = [50, 1]#, 1e-4]
#ub = [900, 6]#, 1]
#
## do particle swarm opti.
#theta0, obj = pso(eq.LLK_rho, lb, ub, args = const, maxiter = 100, swarmsize = 100)
#
## plots
#f, ax = plt.subplots(1, figsize = (7,4))
#
#theta1 = np.array([290.4, 5.2])
#ax.plot(r, densities, 'o') #, alpha = 0.6, color = 'k')
#rplot = np.linspace((rmin),(rmax),500)
#ax.plot(rplot, (eq.rho(rplot, rho0, theta0[0], theta0[1])),'-',color='r')
#ax.plot(rplot, (eq.rho(rplot, rho0, theta1[0], theta1[1])),'-',color='b')
#for be in bin_edges:
#    ax.axvline(be,color='k',linestyle=':')
#ax.set_xscale('log')
#ax.set_yscale('log')


    
print(datetime.now().timestamp() - start.timestamp())

