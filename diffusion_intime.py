# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:42:22 2019

@author: enic156
"""

import eq_functions as eq
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from datetime import datetime
from pyswarm import pso
from scipy import optimize
import emcee

np.random.seed(1756)
random.seed(1756)

start = datetime.now()

fname = 'newberry.txt'
f = open(fname, 'r')
flines = f.readlines()
f.close()

# set up dataframe for dataset/location specific metrics
rr = [42.088309, -113.387377] # 42.087593, -113.387119 ; 42.087905, -113.390624
nwb = [43.726077, -121.309651] #  43.726066, -121.310861 ; 43.726208, -121.309869 ; 43.726179, -121.309841 ; 43.726077, -121.309651
metrics = pd.DataFrame({'lat0':[nwb[0], rr[0]],
                        'long0':[nwb[1], rr[1]],
                        't0':[datetime(2012, 10, 29, hour=8, minute=2, second=21), datetime(2010, 10, 2 , hour=8, minute=13, second=26)],
                        'ru':[10**2.63, 10**3.2],
                        'rl':[3*1e1, 150],
                        'year':['2014','2016']},
    index =  ['newberry.txt','raft_river.txt'])
r = 6371e3 # earth radius in m
lat0, long0, t0 =  metrics.loc[fname].lat0, metrics.loc[fname].long0, metrics.loc[fname].t0 #43.725820, -121.310371 OG # newberry # 42.088047, -113.385184# raft river #  
x0 = eq.gnom_x(lat0,long0,lat0,long0)
y0 = eq.gnom_y(lat0,long0,lat0,long0)

# store events in list
# latitudes and longitudes are projected to a plane using a gnomonic projection
# distances between points and the origin (mean lat,lon) are determined by great circle distances between origin and the points
events_all = [eq.Event(float(event.split()[10]), \
                       (datetime(int(event.split()[0]),int(event.split()[2]),int(event.split()[3]),hour=int(event.split()[4]),minute=int(event.split()[5]),second=int(float(event.split()[6])))-t0).total_seconds(), \
                       r*eq.gnom_x(float(event.split()[7]),float(event.split()[8]),lat0,long0, deg = True),\
                       r*eq.gnom_y(float(event.split()[7]),float(event.split()[8]),lat0,long0, deg = True),\
                       '-', eq.gcdist(r,float(event.split()[7]),lat0,float(event.split()[8]),long0, deg = True), 0) for event in flines]# if event.split()[0] in year]

# format events in the pd dataframe format defined by generate_catalog etc. 
catalog0 = pd.DataFrame({'Magnitude': [event.magnitude for event in events_all],
                                   'Events':'-',
                                   'n_avg':'-',
                                   'Time':[event.time for event in events_all],
                                   'Distance':['-'] * len(events_all),
                                   'x':[event.x for event in events_all],
                                   'y':[event.y for event in events_all],
                                   'Generation':[0] * len(events_all),
                                   'Distance_from_origin': [event.distance_from_origin for event in events_all],
                                   'Year':[event.split()[0] for event in flines],
                                   'Date':[datetime(int(event.split()[0]),int(event.split()[2]),int(event.split()[3]),hour=int(event.split()[4]),minute=int(event.split()[5]),second=int(float(event.split()[6]))) for event in flines]})
cols = ['n_avg','Events','Magnitude','Generation','x','y','Distance','Time','Distance_from_origin','Year','Date']
catalog0 = catalog0.reindex(columns = cols)
catalog0 = catalog0[catalog0.Year == metrics.loc[fname].year]
catalog0 = catalog0[:251]
N = len(catalog0)

nsplit = 4
amount = np.ceil(np.linspace(N/nsplit, N, nsplit))
f, ax = plt.subplots(1, figsize = (7,4))
for i, n in enumerate(amount):
    catalog = catalog0[:int(n)] # only get first injection round 
    k = 22
#    eq.plot_catalog(catalog, 1, np.array([0,0]), color = 'Generation', k = k, saveplot = False, savepath = fname.split(sep='.')[0]+'_positions.png')
    r, densities = eq.plot_ED(catalog, k = k,  plot = False) # get distance, density
    
    # estimate densities prior to filtering by distance, so that they are not affected by absent events
    mask_filter_farevents =  r < metrics.loc[fname].ru # mask to get rid of very far field event
    r = r[mask_filter_farevents]
    densities = densities[mask_filter_farevents]
    
    #mask_fitregion = r > metrics.loc[fname].rl
    catalog.Time = catalog.Time - catalog.Time.min() # adjust so that first event is at time 0
    
    # David's models
    #==============================================================================
    # minimise weighted sum of squares
    #rho0 = np.mean(densities[0:10])
    rmax = (r.max())
    rmin = (r.min())
    n_edges = 32
    bin_edges = np.linspace(np.log10(rmin), np.log10(rmax), n_edges) #np.array([r[i] for i in range(0, len(r), q)])
    bin_edges = 10**bin_edges
    #bin_edges = np.linspace(rmin, rmax, n_edges) #np.array([r[i] for i in range(0, len(r), q)])
    t_now = catalog.Time.max()
    #rc = metrics.loc[fname].rl
    #rc = 50
    
    
    # bounds for the NEWBERRY case
    lb = [5.9e-14, 2.246e+6-1, 1e-15, 0.8e-6, 10, t_now-1, 10]
    ub = [6.1e-1, 2.246e+6+1, 1e-7, 1.3e-6, 1000, t_now+1, 110]
    # alpha, T, k, nu, q, t_now, rc
    bounds = [(low, high) for low, high in zip(lb,ub)] # basinhop bounds
    const = (r, densities, bin_edges, False, lb, ub)
    
    # do particle swarm opti.
    theta0, obj = pso(eq.robj_diff, lb, ub, args = const, maxiter = 100, swarmsize = 2000, phip = 0.75, minfunc = 1e-12, minstep = 1e-12, phig = 0.8)#, f_ieqcons = eq.con_diff)
    #theta_guess = theta0
    #minimizer_kwargs = {"args":const, "bounds":bounds}#, "method":"L-BFGS-B"}
    #annealed = optimize.basinhopping(eq.robj_diff, theta_guess, minimizer_kwargs = minimizer_kwargs, niter = 1000)
    #theta0 = annealed.x
    
    # plots
#    f, ax = plt.subplots(1, figsize = (7,4))
    alpha, T, k, nu, q, t_now, rc = theta0
    
    ax.plot(r, densities, 'o', alpha = 0.5) 
    #ax.set_ylim(ax.get_ylim())
    rplot = np.linspace((rmin),(rmax),500)
    ax.plot(rplot, eq.p2D_transient(rplot, t_now, alpha, T, k, nu, q, rc),'-')
    ax.set_xscale('log')
    ax.set_yscale('log')
plt.title(fname.split(sep=".")[0])
    #==============================================================================

print(datetime.now() - start)