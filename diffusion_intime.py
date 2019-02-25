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

fname = 'raft_river.txt'
f = open(fname, 'r')
flines = f.readlines()
f.close()

# set up dataframe for dataset/location specific metrics
rr = [42.088309, -113.387377] # 42.087593, -113.387119 ; 42.087905, -113.390624
nwb = [43.726077, -121.309651] #  43.726066, -121.310861 ; 43.726208, -121.309869 ; 43.726179, -121.309841 ; 43.726077, -121.309651
brd = [39.791999, -119.013119]
rrt0 = datetime(2010, 10, 2 , hour=8, minute=13, second=26) # first event times
nwbt0 = datetime(2012, 10, 29, hour=8, minute=2, second=21)
brdt0 = datetime(2010, 11, 13 , hour=8, minute=37, second=24)
metrics = pd.DataFrame({'lat0':[nwb[0], rr[0], brd[0]],
                        'long0':[nwb[1], rr[1], brd[1]],
                        't0':[nwbt0, rrt0, brdt0],
                        'ru':[10**2.63, 10**3.2, 1*1e3],
                        'rl':[3*1e1, 150, 1e1],
                        'T_inj':[1.901e+6, 1.616e+7, 1.642e+6],
                        'T_adjust':[432000, 1.382e+6, 0],
                        'range':[slice(150,402),slice(48,82), slice(0,-1)],
                        'q_lwr':[10,16.8,10],
                        'q_upr':[1000,883.2,1000],
                        'rc_lwr':[40,320,50],
                        'rc_upr':[100,450,300]},
    index =  ['newberry.txt','raft_river.txt','bradys.txt'])
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
                                   'Time':[event.time for event in events_all],
                                   'x':[event.x for event in events_all],
                                   'y':[event.y for event in events_all],
                                   'Generation':[0] * len(events_all),
                                   'Distance_from_origin': [event.distance_from_origin for event in events_all],
                                   'Date':[datetime(int(event.split()[0]),int(event.split()[2]),int(event.split()[3]),hour=int(event.split()[4]),minute=int(event.split()[5]),second=int(float(event.split()[6]))) for event in flines]})
cols = ['Magnitude','Generation','x','y','Time','Distance_from_origin','Date']
catalog0 = catalog0.reindex(columns = cols)
catalog0 = catalog0[metrics.loc[fname].range] # get events from time period of interest
catalog0.Time = catalog0.Time - catalog0.Time.min() + metrics.loc[fname].T_adjust # shift time so that first event occurs however long after injection began 
N = len(catalog0)
k = 22
eq.plot_catalog(catalog0, 1, np.array([0,0]), color = 'Generation', k = k, saveplot = False, savepath = fname.split(sep='.')[0]+'_positions.png')

nsplit = 1
amount = np.ceil(np.linspace(N/nsplit, N, nsplit))
#amount = np.array([150,180,251])
r_all = []
dens_all = []
t = []
T = metrics.loc[fname].T_inj
for i, n in enumerate(amount):
    catalog = catalog0.copy()
    catalog = catalog[:int(n)] # only get first injection round 
    assert len(catalog) > k, "Number of nearest neighbours exceed catalog size"
    r, densities = eq.plot_ED(catalog, k = k,  plot = False) # get distance, density
    
    # estimate densities prior to filtering by distance, so that they are not affected by absent events
    mask_filter_farevents =  r < metrics.loc[fname].ru # mask to get rid of very far field events
    r = r[mask_filter_farevents]
    densities = densities[mask_filter_farevents]
#    densities *= 100
    r_all.append(r)
    dens_all.append(densities)
    
    #mask_fitregion = r > metrics.loc[fname].rl
#    catalog.Time = catalog.Time - catalog.Time.min() # adjust so that first event is at time 0
    
    # David's models
    #==============================================================================
    # minimise weighted sum of squares
    #rho0 = np.mean(densities[0:10])
    rmax = (r.max())
    rmin = (r.min())
    t_now = catalog.Time.max()
    t.append(t_now)
    #rc = metrics.loc[fname].rl
    #rc = 50
    
    
# bounds for the NEWBERRY case
lb = [1e-16, 1e-14, 0.8e-6, metrics.loc[fname].q_lwr, metrics.loc[fname].rc_lwr, 1e-3]+[1e-5]*nsplit
ub = [0.1, 1e-6, 1.3e-6, metrics.loc[fname].q_upr, metrics.loc[fname].rc_upr, 1e1]+[1e1]*nsplit
# alpha, k, nu, q, rc, pc, C
bounds = [(low, high) for low, high in zip(lb,ub)] # basinhop bounds
const = (r_all, dens_all, False, lb, ub, T, t)

# do particle swarm opti.
theta0, obj = pso(eq.robj_diff, lb, ub, args = const, maxiter = 500, swarmsize = 1000, phip = 0.75, minfunc = 1e-12, minstep = 1e-12, phig = 0.8)#, f_ieqcons = eq.con_diff)
#theta_guess = theta0
#minimizer_kwargs = {"args":const, "bounds":bounds}#, "method":"L-BFGS-B"}
#annealed = optimize.basinhopping(eq.robj_diff, theta_guess, minimizer_kwargs = minimizer_kwargs, niter = 1000)
#theta0 = annealed.x

# plots
#    f, ax = plt.subplots(1, figsize = (7,4))
alpha, k, nu, q, rc, pc = theta0[:6]
C = theta0[6:]
colors = ['r','b','y','k']
f, ax = plt.subplots(1, figsize = (7,4))
rplot = np.linspace((rmin),(rmax),500)
rplot_all = [rplot, rplot, np.linspace(rmin,rmax,500)]
for i, n in enumerate(amount):
    ax.plot(r_all[i], dens_all[i], 'o', alpha = 0.3, color = colors[i])
    dens_model = eq.p2D_transient(rplot_all[i], t[i], C[i], pc, alpha, T, k, nu, q, rc)
#    dens_model[dens_model<=0] = np.nan
    ax.plot(rplot_all[i], dens_model,'-',label='At {0:.1f} days'.format(t[i]/60/60/24), color = colors[i])
#ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_xlabel('distance from well (m)')
ax.set_ylabel(r'event density $(/m^2)$')
plt.title(fname.split(sep=".")[0])
plt.legend(loc = 'upper right')
#plt.savefig('diff_time_3split_loglog.png',dpi=400)
##==============================================================================

## generate synthesis plot
#catalogPlots = catalog0.copy()
#fflow, axflow = plt.subplots(2, figsize=(10,5), gridspec_kw = {'height_ratios':[3, 1]})
#t = np.array(catalogPlots.Time) # get times of events
##t = t#/60/60/24
#axflow[0].hist(t, bins=40, color='r', edgecolor='k')
#axflow[0].set(ylabel = 'Events', title = fname, xticklabels = [])
#axflow[0].set_xlim(0, t.max())
##plt.tight_layout()
##plt.savefig('newberr_event_freq.png',dpi=400)
#
##f3, ax3 = plt.subplots(1, figsize=(8,2))
#t_inject = np.linspace(0,t.max(),100)
##injection = lambda t: np.array([q/100 if ti < T/60/60/24 else 0 for ti in t_inject])
#flow = np.array([q/100 if ti < T else 0 for ti in t_inject])
#axflow[1].plot(t_inject, flow)
#axflow[1].set(xlabel='Time since injection begun (s)', ylabel = r'Flwo Rate $l/s$',xlim=(0, t.max()),ylim=(-0.1,flow.max()+1))
#
#plt.tight_layout()
#plt.savefig(fname.split(sep='.')[0]+'_flowandfreq.png',dpi=400)


print(datetime.now() - start)