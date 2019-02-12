# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:22:02 2019

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

fname = 'newberry.txt'
f = open(fname, 'r')
flines = f.readlines()

# set up dataframe for dataset/location specific metrics
rr = [42.088309, -113.387377] # 42.087593, -113.387119 ; 42.087905, -113.390624
nwb = [43.726077, -121.309651] #  43.726066, -121.310861 ; 43.726208, -121.309869 ; 43.726179, -121.309841 ; 43.726077, -121.309651
metrics = pd.DataFrame({'lat0':[nwb[0], rr[0]],
                        'long0':[nwb[1], rr[1]],
                        't0':[datetime(2012, 10, 29, hour=8, minute=2, second=21), datetime(2010, 10, 2 , hour=8, minute=13, second=26)],
                        'ru':[10**2.7, 10**3.5],
                        'rl':[10**1.2, 10**1.8],
                        'year':['2014','2016']},
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
catalog = catalog[catalog.Year == metrics.loc[fname].year]
catalog = catalog[(catalog.Distance_from_origin < metrics.loc[fname].ru)]

N = len(catalog)
k = 20
eq.plot_catalog(catalog, 1, np.array([0,0]), color = 'Generation', k = k, saveplot = False, savepath = fname.split(sep='.')[0]+'_positions.png')
r, densities = eq.plot_ED(catalog, k = k,  plot = False) # get distance, density

# David's models
#==============================================================================
## perform particle swarm optimisation (MLE)
#rho0 = np.mean(densities[0:100])
#rmax = (r.max())
#rmin = (r.min())
#n_edges = 32
#bin_edges = np.linspace(np.log10(rmin), np.log10(rmax), n_edges) #np.array([r[i] for i in range(0, len(r), q)])
#bin_edges = 10**bin_edges
##bin_edges = np.linspace(rmin, rmax, n_edges) #np.array([r[i] for i in range(0, len(r), q)])
#T = 2e6
##q = 5e-6
#const = (rmax, rmin, r, bin_edges, n_edges, T)
#
#lb = [1e-3, 1e-16, 1e-2, 1e-7]
#ub = [1e-1, 1e-3, 4e-1  , 4e-4]
## alpha, k, nu, q= theta
#
## do particle swarm opti.
#theta0, obj = pso(eq.llk_diff, lb, ub, args = const, maxiter = 100, swarmsize = 100)
#
## plots
#f, ax = plt.subplots(1, figsize = (7,4))
#
#ax.plot(r, densities, 'o') 
##ax.set_ylim(ax.get_ylim())
#rplot = np.linspace((rmin),(rmax),500)
#ax.plot(rplot, eq.p3D(rplot, True, T, theta0[0], theta0[1], theta0[3], theta0[2])+1e-3,'-',color='r')
#for be in bin_edges:
#    ax.axvline(be,color='k',linestyle=':')
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.set_title(fname.split(sep=".")[0])
#==============================================================================
#==============================================================================
# perform particle swarm optimisation (least squares)
#rho0 = np.mean(densities[0:100])
rmax = (r.max())
rmin = (r.min())
#n_edges = 32
#bin_edges = np.linspace(np.log10(rmin), np.log10(rmax), n_edges) #np.array([r[i] for i in range(0, len(r), q)])
#bin_edges = 10**bin_edges
#bin_edges = np.linspace(rmin, rmax, n_edges) #np.array([r[i] for i in range(0, len(r), q)])
T = 1e7
const = (r, densities)

lb = [1e-3, 1e-12, 1e-2, 1e-7, 1e3]
ub = [1e-1, 1e-3, 4e-1  , 4e-4, 1e8]
#alpha, k, nu, q = theta

# do particle swarm opti.
theta0, obj = pso(eq.robj_diff, lb, ub, args = const, maxiter = 500, swarmsize = 1000)

# plots
f, ax = plt.subplots(1, figsize = (7,4))

ax.plot(r, densities, 'o') 
#ax.set_ylim(ax.get_ylim())
rplot = np.linspace((rmin),(rmax),500)
ax.plot(rplot, eq.p3D(rplot, True, theta0[4], theta0[0], theta0[1], theta0[3], theta0[2]),'-',color='r')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title(fname.split(sep=".")[0])
#==============================================================================







#==============================================================================
## perform particle swarm optimisation on Goebel/Brodsky model (least squares obj)
#rmax = r.max()
#rmin = r.min()
#n_edges = 10
#bin_edges = np.linspace(np.log10(rmin), np.log10(rmax), n_edges) #np.array([r[i] for i in range(0, len(r), q)])
#bin_edges = 10**bin_edges
##bin_edges = np.linspace(rmin, rmax, n_edges) #np.array([r[i] for i in range(0, len(r), q)])
#const = (r, densities, bin_edges)
#
#lb = [1, 1, 1e-4]
#ub = [1000, 6, 1]
#
## do particle swarm opti.
#theta0, obj = pso(eq.robj, lb, ub, args = const, maxiter = 100, swarmsize = 500)
#
#f, ax = plt.subplots(1, figsize = (7,4))
#ax.plot(r, densities, 'o', alpha = 0.3)
#rplot = np.linspace((rmin),(rmax),500)
#ax.plot(rplot, (eq.rho(rplot, theta0[2], theta0[0], theta0[1])),'-')
#ax.set_title(fname.split(sep=".")[0]+" "+metrics.loc[fname].year)
##for be in bin_edges:
##    ax.axvline(be,color='k',linestyle=':')
#ax.set_xscale('log')
#ax.set_yscale('log')
#==============================================================================
## perform particle swarm optimisation (MLE)
#rho0 = np.mean(densities[0:100])
#rmax = (r.max())
#rmin = (r.min())
#n_edges = 32
#bin_edges = np.linspace(np.log10(rmin), np.log10(rmax), n_edges) #np.array([r[i] for i in range(0, len(r), q)])
#bin_edges = 10**bin_edges
##bin_edges = np.linspace(rmin, rmax, n_edges) #np.array([r[i] for i in range(0, len(r), q)])
#const = (rmax, rmin, r, bin_edges, n_edges, rho0)
#
#lb = [50, 1]
#ub = [900, 6]
#
## do particle swarm opti.
#theta0, obj = pso(eq.LLK_rho, lb, ub, args = const, maxiter = 100, swarmsize = 100)
#
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
#ax.set_title(fname.split(sep=".")[0])
#==============================================================================

print(datetime.now() - start)