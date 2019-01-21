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
from scipy import stats
from datetime import datetime
from pyswarm import pso

np.random.seed(1756)
random.seed(1756)

start = datetime.now()

k = 2000 # max number of events considered
fname = 'newberry.txt'
f = open(fname, 'r')
flines = f.readlines()
if len(flines) > k:
    # reduce events to a random sample of k elements
    flines = random.sample(flines, k)

f.close()
# get longitudes and latitudes
lats = [float(event.split()[7]) for event in flines]
longs = [float(event.split()[8]) for event in flines]
lat_mean = np.mean(lats)
long_mean = np.mean(longs)
r = 6.3781e6 # earth radius in m

# store events in list
# NOTES:
# latitude and longitude are converted to rectangular coordinates by
# projecting the sines of the lat. and long. on to the plane tangent to the point of mean lat. and long.
# distances between points and the origin (mean lat,lon) are determined by great circle distances between origin and the points
events_all = [eq.Event(float(event.split()[10]), '-', r*np.sin(np.pi/180*(float(event.split()[8])-long_mean)), r*np.sin(np.pi/180*(float(event.split()[7])-lat_mean)), '-', eq.gcdist(r,float(event.split()[7]),lat_mean,float(event.split()[8]),long_mean), 0) for event in flines]

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

eq.plot_catalog(catalog, 1, np.array([0,0]), color = 'Generation')
distances, densities = eq.plot_ED(catalog, plot = False) # get distance, density

# need to be in log space
#distances = np.log10(distances)
#densities = np.log10(densities)

# perform particle swarm optimisation in parameter space on log likelihood
rho0 = np.mean(densities[0:6])
scale = distances.min() # scale distances so that min dist. is 1
r = distances
rmax = r.max()
const = (rho0, rmax, r)

lb = [1e-3, 1e-3]
ub = [r.max(), 15]

#vmin = 1e-5
#vmax = 15
#
#f, ax = plt.subplots(1, figsize = (7,7))
#n = 100
#rc = np.linspace(1, r.max(), n)
#gmma = np.linspace(1, 10, n)
#X, Y = np.meshgrid(rc, gmma)
#Z = np.zeros(np.shape(X))
#for i in range(n):
#    for j in range(n):
#        Z[i][j] = eq.LLK_rho([X[i][j],Y[i][j]], rho0, rmax, r)
#cs = plt.contourf(X,Y,Z,200)
#f.colorbar(cs, ax=ax)
#plt.show()

theta0, llk0 = pso(eq.LLK_rho, lb, ub, args = const, maxiter = 100, swarmsize = 120)

f2, ax2 = plt.subplots(1, figsize = (7,7))

ax2.plot(distances, densities, 'o')

rplot = np.linspace(distances.min(),distances.max(),500)
ax2.plot(rplot, eq.rho(rplot, rho0, theta0[0], theta0[1]),'-',color='r')
ax2.set_xscale('log')
ax2.set_yscale('log')

print(datetime.now() - start)