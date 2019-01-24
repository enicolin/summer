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

k = 2000 # max number of events considered
fname = 'raft_river.txt'
f = open(fname, 'r')
flines = f.readlines()
if len(flines) > k:
    # reduce events to a random sample of k elements
    flines = random.sample(flines, k)

f.close()
# get longitudes and latitudes
lats = [float(event.split()[7]) for event in flines]
longs = [float(event.split()[8]) for event in flines]

r = 6.3781e6 # earth radius in m
lat0 =  42.087422 ## raft river
long0 = -113.392872## raft river
x0 = eq.gnom_x(lat0,long0,lat0,long0)
y0 = eq.gnom_y(lat0,long0,lat0,long0)
#lat0 = np.mean(lats)
#long0 = np.mean(longs)

# store events in list
# NOTES:
# latitudes and longitudes are projected to a plane using a gnomonic projection
# distances between points and the origin (mean lat,lon) are determined by great circle distances between origin and the points
year = '2016'
events_all = [eq.Event(float(event.split()[10]), '-', \
                       r*eq.gnom_x(float(event.split()[7]),float(event.split()[8]),lat0,long0, deg = True),\
                       r*eq.gnom_y(float(event.split()[7]),float(event.split()[8]),lat0,long0, deg = True),\
                       '-', eq.gcdist(r,float(event.split()[7]),lat0,float(event.split()[8]),long0, deg = True), 0) for event in flines]# if event.split()[0] == year]

# format events in the pd dataframe format defined by generate_catalog etc. 
catalog = pd.DataFrame({'Magnitude': [event.magnitude for event in events_all],
                                   'Events':'-',
                                   'n_avg':'-',
                                   'Time':[0] * len(events_all),
                                   'Distance':['-'] * len(events_all),
                                   'x':[event.x for event in events_all],
                                   'y':[event.y for event in events_all],
                                   'Generation':[0] * len(events_all),
                                   'Distance_from_origin': [event.distance_from_origin for event in events_all]})
cols = ['n_avg','Events','Magnitude','Generation','x','y','Distance','Time','Distance_from_origin']
catalog = catalog.reindex(columns = cols)
catalog = catalog[catalog.Magnitude <= 6]
N = len(events_all)

eq.plot_catalog(catalog, 1, np.array([x0,y0]), color = 'Generation')
distances, densities = eq.plot_ED(catalog, k = int(N**0.5),  plot = False) # get distance, density

# perform particle swarm optimisation in parameter space on log likelihood
rho0 = np.mean(densities[0:5])
scale = 1#distances.max() # scale distances so that max dist. is 1
r = distances/scale
rmax = r.max()
rmin = r.min()
#q = 2
n_edges = 300
bin_edges = np.linspace(r.min(), r.max(), n_edges) #np.array([r[i] for i in range(0, len(r), q)])
const = (rmax, rmin, r, rho0, bin_edges, n_edges)

lb = [1, 1]
ub = [2500, 5]

#f, ax = plt.subplots(1, figsize = (7,7))
#n = 125
#rc = np.linspace(100, 500, n)
#gmma = np.linspace(1, 5, n)
#X, Y = np.meshgrid(rc, gmma)
#Z = np.zeros(np.shape(X))
#for i in range(n):
#    for j in range(n):
#        Z[i][j] = eq.LLK_rho([X[i][j],Y[i][j]], rmax, rmin, r, rho0, bin_edges, q)
#cs = plt.contourf(X,Y,Z,80,colormap = 'plasma')
#f.colorbar(cs, ax=ax)
#plt.show()

theta0, llk0 = pso(eq.LLK_rho, lb, ub, args = const, maxiter = 100, swarmsize = 200)
# by eye, good fit given by rc = 320, gmma = 1.8

f2, ax2 = plt.subplots(1, figsize = (7,7))

ax2.plot(distances/scale, densities, 'o')

rplot = np.linspace(rmin,rmax,500)
ax2.plot(rplot, eq.rho(rplot, rho0, theta0[0], theta0[1]),'-',color='r')
ax2.set_xscale('log')
ax2.set_yscale('log')

print(datetime.now() - start)
