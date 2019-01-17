# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:08:31 2019

@author: enic156
"""
import eq_functions as eq
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

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

# take bootstrap samples and plot data

K = 20
N = int(len(events_all))
# prepare meshgrid
npoints = 100j
npoint_real = int(np.linalg.norm(npoints))
Z = np.zeros((npoint_real,npoint_real))

print("EVENT POSITIONS AND DENSITY FOR {}".format(fname.split(sep='.')[0]).upper())

# perform initial kNN density computation on whole set in order to get meshgrid bounds
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
distance_all, density_all = eq.plot_ED(catalog, plot = True) # get distance, density
dist_min = np.log(distance_all.min())
dist_max = np.log(distance_all.max())
density_min = np.log(density_all.min())
density_max = np.log(density_all.max()) 

for k in np.arange(K):
    events = np.random.choice(events_all,N)
    # format events in the pd dataframe format defined by generate_catalog etc. 
    catalog = pd.DataFrame({'Magnitude': [event.magnitude for event in events],
                                       'Events':'-',
                                       'n_avg':'-',
                                       'Time':[0] * len(events),
                                       'Distance':['-'] * len(events),
                                       'x':[event.x for event in events],
                                       'y':[event.y for event in events],
                                       'Generation':[0] * len(events),
                                       'Distance_from_origin': [event.distance_from_origin for event in events]})
    cols = ['n_avg','Events','Magnitude','Generation','x','y','Distance','Time','Distance_from_origin']
    catalog = catalog.reindex(columns = cols)
    catalog = catalog[catalog.Magnitude <= 6]
    
    #eq.plot_catalog(catalog, 2, np.array([0,0]), color = 'Generation')
    
    distance, density = eq.plot_ED(catalog, plot = False) # get distance, density
    
    distance, density = np.log(distance), np.log(density)
    
    X, Y = np.mgrid[dist_min:dist_max:npoints, density_min:density_max:npoints]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([distance, density])
    kernel = stats.gaussian_kde(values)
    Z += np.reshape(kernel(positions).T, X.shape)


Z /= K # divide through by num bootstrap samples

f, ax = plt.subplots(1, figsize = (7,6))
cs = plt.contourf(X, Y, Z, 50, cmap = 'RdYlGn')
f.colorbar(cs, ax=ax)
plt.plot(np.log(distance_all),np.log(density_all),'o',alpha=0.5) # also plot entire data set
#ax.set_yscale('log')
ax.set_xlabel('distance from main shock')
ax.set_ylabel('event density')
ax.set_ylim(0,density.max())
#ax.set_xscale('log')
plt.xlim([dist_min,dist_max])
plt.ylim([density_min, density_max])
plt.show()

print(datetime.now() - start)