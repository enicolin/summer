# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 08:26:39 2019

@author: enic156
"""

import eq_functions as eq
import numpy as np
import matplotlib.pyplot as plt
from math import log
import pandas as pd

#==============================================================================
# a = np.random.uniform(1,5)
# b = np.random.uniform(1,5)
# c = np.random.uniform(1,5)
# 
# r = np.linspace(0,10,75)
# y1 = eq.rho(r, a, b, c)
# y2 = eq.rho2(r, a, b, c)
# 
# f, ax = plt.subplots(1, figsize = (7,7))
# 
# ax.plot(r, y1, color = 'r')
# ax.plot(r, y2, 'o', color = 'k')
# ax.set_xscale('log')
# ax.set_yscale('log')
#==============================================================================

#==============================================================================
# 
# def logfactrue(n):
#     return np.array([np.log(float(np.math.factorial(ni))) for ni in n])
# 
# def logfacappr(n):
#     return n*np.log(n) - n + 1
# 
# k = 100
# n = np.linspace(1,k,k)
# y1 = logfactrue(n)
# y2 = logfacappr(n)
# 
# f, ax = plt.subplots(1, figsize = (7,7))
# ax.plot(n, y1, color = 'r', label = 'true')
# ax.plot(n, y2, color = 'b', label = 'approx')
#==============================================================================

#==============================================================================
# ax.legend()
# 
# plt.show()
#==============================================================================
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

#eq.plot_catalog(catalog, 1, np.array([0,0]), color = 'Generation')
distances, densities = eq.plot_ED(catalog, plot = False) # get distance, density
q = 4
bin_edges = np.array([r[i] for i in range(0, len(r), q)])

theta0 = np.array([320.32, 1.8])
const = (distances.max(), distances.min(), distances, np.mean(densities[0:6]), bin_edges, q)

a = eq.LLK_rho(theta0, const[0], const[1], const[2], const[3], const[4], const[5])
