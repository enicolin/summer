# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:16:44 2019

@author: enic156
"""

import eq_functions as eq
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from datetime import datetime
from pyswarm import pso

np.random.seed(1756)
random.seed(1756)

start = datetime.now()

#k = 2000 # max number of events considered
fname = 'newberry.txt'
f = open(fname, 'r')
flines = f.readlines()
#if len(flines) > k:
#    # reduce events to a random sample of k elements
#    flines = random.sample(flines, k)

f.close()
rr = [42.087593, -113.387119]
nwb = [43.726179, -121.309841]
metrics = pd.DataFrame({'lat0':[nwb[0], rr[0]],
                        'long0':[nwb[1], rr[1]],
                        't0':[datetime(2012, 10, 29, hour=8, minute=2, second=21), datetime(2010, 10, 2 , hour=8, minute=13, second=26)]},
    index =  ['newberry.txt','raft_river.txt'])
r = 6371e3 # earth radius in m
lat0, long0, t0 =  metrics.loc[fname].lat0, metrics.loc[fname].long0, metrics.loc[fname].t0 #43.725820, -121.310371 OG # newberry # 42.088047, -113.385184# raft river #  
x0 = eq.gnom_x(lat0,long0,lat0,long0)
y0 = eq.gnom_y(lat0,long0,lat0,long0)

# store events in list
# NOTES:
# latitudes and longitudes are projected to a plane using a gnomonic projection
# distances between points and the origin (mean lat,lon) are determined by great circle distances between origin and the points
events_all = [eq.Event(float(event.split()[10]), \
                       (datetime(int(event.split()[0]),int(event.split()[2]),int(event.split()[3]),hour=int(event.split()[4]),minute=int(event.split()[5]),second=int(float(event.split()[6])))-t0).total_seconds(), \
                       r*eq.gnom_x(float(event.split()[7]),float(event.split()[8]),lat0,long0, deg = True),\
                       r*eq.gnom_y(float(event.split()[7]),float(event.split()[8]),lat0,long0, deg = True),\
                       '-', eq.gcdist(r,float(event.split()[7]),lat0,float(event.split()[8]),long0, deg = True), 0) for event in flines]# if event.split()[0] in year]

# format events in the pd dataframe format defined by generate_catalog etc. 
catalog = pd.DataFrame({'Magnitude': [event.magnitude for event in events_all],
                                   'Events':'-',
                                   'n_avg':'-',
                                   'Time':[event.time for event in events_all],
                                   'Distance':['-'] * len(events_all),
                                   'x':[event.x for event in events_all],
                                   'y':[event.y for event in events_all],
                                   'Generation':[0] * len(events_all),
                                   'Distance_from_origin': [event.distance_from_origin for event in events_all],
                                   'Year':[event.split()[0] for event in flines]})
cols = ['n_avg','Events','Magnitude','Generation','x','y','Distance','Time','Distance_from_origin','Year']
catalog = catalog.reindex(columns = cols)
#catalog = catalog[catalog.Year == '2016'] # for RR
N = len(catalog)
k = 6 #int(N**0.5)

eq.plot_catalog(catalog, 1, np.array([0,0]), color = 'Generation', k = k, saveplot = True, savepath = fname.split(sep='.')[0]+'_positions.png')

r, densities = eq.plot_ED(catalog, k = k,  plot = False) # get distance, density
df_dens = pd.DataFrame({'distance':r, 'density':densities})
df_dens = df_dens[(df_dens.distance > 10**1.2) & (df_dens.distance < 10**2.5)]
r = df_dens.distance
densities = df_dens.density


# perform particle swarm optimisation in parameter space on log likelihood
rho0 = np.mean(densities[0:6])
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
#theta0, obj = pso(eq.robj, lb, ub, args = const, maxiter = 500, swarmsize = 500)

# plots
f, ax = plt.subplots(1, figsize = (7,7))

theta0 = np.array([212.8, 4.4, rho0])
ax.plot(r, densities, 'o', alpha = 0.6, color = 'k')
rplot = np.linspace((rmin),(rmax),500)
ax.plot(rplot, (eq.rho(rplot, theta0[2], theta0[0], theta0[1], plot = True)),'-',color='b')
for be in bin_edges:
    ax.axvline(be,color='k',linestyle=':')
ax.set_xscale('log')
ax.set_yscale('log')


print(datetime.now() - start)


