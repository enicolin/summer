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
fname = 'newberry.txt'
f = open(fname, 'r')
flines = f.readlines()
if len(flines) > k:
    # reduce events to a random sample of k elements
    flines = random.sample(flines, k)

f.close()

r = 6371e3 # earth radius in m
lat0, long0 = 43.725820, -121.310371 # newberry # 42.088047, -113.385184# raft river # 
x0 = eq.gnom_x(lat0,long0,lat0,long0)
y0 = eq.gnom_y(lat0,long0,lat0,long0)

# store events in list
# NOTES:
# latitudes and longitudes are projected to a plane using a gnomonic projection
# distances between points and the origin (mean lat,lon) are determined by great circle distances between origin and the points
events_all = [eq.Event(float(event.split()[10]), event.split()[0], \
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
                                   'Distance_from_origin': [event.distance_from_origin for event in events_all]})
cols = ['n_avg','Events','Magnitude','Generation','x','y','Distance','Time','Distance_from_origin']
catalog = catalog.reindex(columns = cols)
catalog = catalog[(catalog.Time == '2014')]# & (catalog.Distance_from_origin <= 10**2.5)]
N = len(catalog)

eq.plot_catalog(catalog, 1, np.array([0,0]), color = 'Generation')

r, densities = eq.plot_ED(catalog, k = 20,  plot = False) # get distance, density

# perform particle swarm optimisation in parameter space on log likelihood
rho0 = np.mean(densities[0:6])
rmax = np.log(r.max())
rmin = np.log(r.min())
n_edges = 250 
#bin_edges = np.linspace(np.log10(rmin), np.log10(rmax), n_edges) #np.array([r[i] for i in range(0, len(r), q)])
#bin_edges = 10**bin_edges
bin_edges = np.linspace(rmin, rmax, n_edges) #np.array([r[i] for i in range(0, len(r), q)])
const = (rmax, rmin, r, rho0, bin_edges, n_edges)

lb = [1, 1]
ub = [400, 6]

# do particle swarm opti.
theta0, llk0 = pso(eq.LLK_rho, lb, ub, args = const, maxiter = 100, swarmsize = 200)

# plots
f, ax = plt.subplots(1, figsize = (7,7))

ax.plot(r, densities, 'o')

rplot = np.linspace(np.exp(rmin),np.exp(rmax),n_edges)
#ax.plot(rplot, (eq.rho(rplot, rho0, 238.6, 2.6)),'-',color='r')
ax.plot(rplot, (eq.rho(rplot, rho0, theta0[0], theta0[1])),'-',color='b')
ax.set_xscale('log')
ax.set_yscale('log')

print(datetime.now() - start)


#f, ax = plt.subplots(1, figsize = (7,7))
#n = 125
#rc = np.linspace(100, 500, n)
#gmma = np.linspace(1, 5, n)
#X, Y = np.meshgrid(rc, gmma)
#Z = np.zeros(np.shape(X))
#for i in range(n):
#    for j in range(n):
#        Z[i][j] = eq.LLK_rho([X[i][j],Y[i][j]], rmax, rmin, r, rho0, bin_edges, n_edges)
#cs = plt.contourf(X,Y,Z,80,colormap = 'plasma')
#f.colorbar(cs, ax=ax)
#plt.show()

#start = datetime.now()
#k = 3
#positions = [np.array([xi,yi]) for xi,yi in zip([event.x for event in events_all], [event.y for event in events_all])]
#f, ax = plt.subplots(1, figsize = (7,7))
#n = 150
#easting = np.linspace(-300, 300, n)
#northing = easting
#X, Y = np.meshgrid(easting, northing)
#Z = np.zeros(np.shape(X))
#for i in range(n):
#    for j in range(n):
#        Z[i][j] = k / (eq.kNN_measure(positions, np.array([easting[i],northing[j]]), k, goebel_dens = True))
#Z = Z.T
#cs = plt.contourf(X,Y,Z,10,cmap = 'coolwarm')
#f.colorbar(cs, ax=ax)
#plt.savefig('newberry_ensity2d.png',dpi=400)
#plt.show()
#print(datetime.now() - start)