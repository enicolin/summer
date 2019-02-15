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

fname = 'newberry.txt'
f = open(fname, 'r')
flines = f.readlines()

# set up dataframe for dataset/location specific metrics
rr = [42.088309, -113.387377] # 42.087593, -113.387119 ; 42.087905, -113.390624
nwb = [43.726077, -121.309651] #  43.726066, -121.310861 ; 43.726208, -121.309869 ; 43.726179, -121.309841 ; 43.726077, -121.309651
metrics = pd.DataFrame({'lat0':[nwb[0], rr[0]],
                        'long0':[nwb[1], rr[1]],
                        't0':[datetime(2012, 10, 29, hour=8, minute=2, second=21), datetime(2010, 10, 2 , hour=8, minute=13, second=26)],
                        'ru':[10**2.8, 10**3.5],
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
catalog = catalog[(catalog.Distance_from_origin > metrics.loc[fname].rl) & (catalog.Distance_from_origin < metrics.loc[fname].ru)]
N = len(catalog)
k = 20 #int(N**0.5)

eq.plot_catalog(catalog, 1, np.array([0,0]), color = 'Generation', k = k, saveplot = True, savepath = fname.split(sep='.')[0]+'_positions.png')

# plots
f, ax = plt.subplots(1, figsize = (7,4))
prts = np.ceil(np.linspace(N/4,N,4))
catalog = catalog.sort_values(by = ['Time'])
for m in prts:
    r, densities = eq.plot_ED(catalog[:int(m)], k = k,  plot = False) # get distance, density
#    densities *= 10
    df_dens = pd.DataFrame({'distance':r, 'density':densities})
    r = np.array(df_dens.distance)
    densities = np.array(df_dens.density)
    #r = np.log10(r)
    #densities = np.log10(densities)
    
    #==============================================================================
    # perform particle swarm optimisation (least squares obj)
    rho0 = np.mean(densities[0:100])
    rmax = r.max()
    rmin = r.min()
    n_edges = 10
    bin_edges = np.linspace(np.log10(rmin), np.log10(rmax), n_edges) #np.array([r[i] for i in range(0, len(r), q)])
    bin_edges = 10**bin_edges
    #bin_edges = np.linspace(rmin, rmax, n_edges) #np.array([r[i] for i in range(0, len(r), q)])
    const = (r, densities, bin_edges)
    
    lb = [1, 1, 1e-4]
    ub = [1000, 6, 1]
    
    # do particle swarm opti.
#    theta0, obj = pso(eq.robj, lb, ub, args = const, maxiter = 100, swarmsize = 500)
    
#        # plots
#    f, ax = plt.subplots(1, figsize = (7,4))
    
    #theta1 = np.array([212.8, 4.4, rho0])
    #    theta1 = np.array([350, 2.6, rho0])
    ax.plot(r, densities, 'o', alpha = 0.3, label = '{:.1f}'.format(m/N*100)+'%')#, color = 'k')
    rplot = np.linspace((rmin),(rmax),500)
    
#    ax.plot(rplot, (eq.rho(rplot, theta0[2], theta0[0], theta0[1])),'-',label=catalog.iloc[int(m-1)].Time)#,color='r')
        #ax.plot(rplot, (eq.rho(rplot, theta1[2], theta1[0], theta1[1])),'-',color='b')
    #    for be in bin_edges:
    #        ax.axvline(be,color='k',linestyle=':')
    #    ax.set_xscale('log')
    #    ax.set_yscale('log')
    #    ax.set_title(fname.split(sep=".")[0]+" "+metrics.loc[fname].year)
    #==============================================================================
ax.legend(title='proportion of data used')
#for be in bin_edges:
#    ax.axvline(be,color='k',linestyle=':')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title(fname.split(sep=".")[0]+" "+metrics.loc[fname].year)

#==============================================================================
## perform particle swarm optimisation (MLE)
#rho0 = np.mean(densities[0:100])
#rmax = (r.max())
#rmin = (r.min())
#n_edges = 26
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
#theta1 = np.array([350, 2.6, rho0])
##theta1 = np.array([212.8, 4.4])
#ax.plot(r, densities, 'o') #, alpha = 0.6, color = 'k')
##ax.set_ylim(ax.get_ylim())
#rplot = np.linspace((rmin),(rmax),500)
#ax.plot(rplot, (eq.rho(rplot, rho0, theta0[0], theta0[1])),'-',color='r')
#ax.plot(rplot, (eq.rho(rplot, rho0, theta1[0], theta1[1])),'-',color='b')
#for be in bin_edges:
#    ax.axvline(be,color='k',linestyle=':')
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.set_title(fname.split(sep=".")[0])
#==============================================================================

print(datetime.now() - start)


