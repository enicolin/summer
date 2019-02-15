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
                        'ru':[10**2.7, 10**3.2],
                        'rl':[10**1.9, 10**1.8],
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
#catalog = catalog[catalog.Year == metrics.loc[fname].year]
catalog = catalog[(catalog.Distance_from_origin < metrics.loc[fname].ru)]# & (catalog.Distance_from_origin > metrics.loc[fname].rl)]

N = len(catalog)
k = 22
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
#const = (rmax, rmin, r, bin_edges, n_edges)
#
#lb = [1e-3, 1e-16, 1e-2, 1e-7, 1]
#ub = [1e-1, 1e-3, 4e-1  , 4e-4, 1e9]
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
#ax.plot(rplot, eq.p3D(rplot, True, theta0[4], theta0[0], theta0[1], theta0[3], theta0[2])+1e-3,'-',color='r')
#for be in bin_edges:
#    ax.axvline(be,color='k',linestyle=':')
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.set_title(fname.split(sep=".")[0])
#==============================================================================
#==============================================================================
## minimise weighted sum of squares
##rho0 = np.mean(densities[0:100])
#rmax = (r.max())
#rmin = (r.min())
#n_edges = 32
#bin_edges = np.linspace(np.log10(rmin), np.log10(rmax), n_edges) #np.array([r[i] for i in range(0, len(r), q)])
#bin_edges = 10**bin_edges
##bin_edges = np.linspace(rmin, rmax, n_edges) #np.array([r[i] for i in range(0, len(r), q)])
#const = (r, densities, bin_edges)
#
##lb = [1e-7, 1e-15, 1e-8, 1e-4, 1e6]
##ub = [1e-5, 1e-7, 1e-5, 1e-3, 1e7]
##alpha, k, nu, q, T  = theta
#lb = [1e-1,1e-5]
#ub = [1e2, 1e7]
## alphaT, knuq
#bounds = [(low, high) for low, high in zip(lb,ub)] # basinhop bounds
#
#
## do particle swarm opti.
#theta0, obj = pso(eq.robj_diff, lb, ub, args = const, maxiter = 100, swarmsize = 500, phip = 0.75, minfunc = 1e-12, minstep = 1e-12, phig = 0.8)
##theta_guess = np.array([  5.84469266e+04,   9.99998849e+05,   8.99666895e+01, 4.87798639e-01,   1.00000000e-09]) # 2D initial guess
##theta_guess = np.array([  20.00000000e-14,   7.68417736e+08,   7.63685917e+01, 4.65534682e-01,   9.04412151e+02]) # 3D initial guess
##theta_guess = np.array([  7.50000000e-11,   19.68417736e+21]) # 3D initial guess
##minimizer_kwargs = {"args":const, "bounds":bounds, "method":"L-BFGS-B"}
##annealed = optimize.basinhopping(eq.robj_diff, theta_guess, minimizer_kwargs = minimizer_kwargs, niter = 10000, T =1e40)
##theta0 = annealed.x
#
## plots
#f, ax = plt.subplots(1, figsize = (7,4))
#
##ax.plot(r, densities2, 'ro') 
#ax.plot(r, densities, 'bo') 
##ax.set_ylim(ax.get_ylim())
#rplot = np.linspace((rmin),(rmax),500)
#ax.plot(rplot, eq.p2D(rplot, theta0[0], theta0[1]),'-',color='r')
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.set_title(fname.split(sep=".")[0])
#==============================================================================







#==============================================================================
# perform particle swarm optimisation on Goebel/Brodsky model (least squares obj)
rmax = r.max()
rmin = r.min()
n_edges = 10
q = 10
bin_edges = np.array([r[i] for i in range(0, len(r), q)]) 
#if len(r) % q != 0:
#    bin_edges = np.concatenate((bin_edges, r[-1:(-(len(r) % q)):-1][::-1])) # append obsevations in last bin that werent enough to fill a bin of q 
const = (r, densities, bin_edges, q, False)

lb = [1, 1, 1e-4]
ub = [1000, 6, 1]

# do particle swarm opti.
theta0, obj = pso(eq.robj, lb, ub, args = const, maxiter = 1000, swarmsize = 500)
#theta0 = np.array([  1.53528973e+02,   2.88098966e+00,   2.33379173e-03])
#theta0 = np.array([  1.19724602e+02,   2.11414066e+00,   3.05878053e-03])

f, ax = plt.subplots(1, figsize = (7,4))
ax.plot(r, densities, 'o', alpha = 0.3)
rplot = np.linspace((rmin),(rmax),500)
ax.plot(rplot, (eq.rho(rplot, theta0[2], theta0[0], theta0[1])),'-')
ax.set_title(fname.split(sep=".")[0]+" "+metrics.loc[fname].year)
#for be in bin_edges:
#    ax.axvline(be,color='k',linestyle=':')
ax.set_xscale('log')
ax.set_yscale('log')

# MCMC sampling

ndim = 3
nwalkers = 32

sampler = emcee.EnsembleSampler(nwalkers, ndim, eq.robj, args = [r, densities, bin_edges, q, True]) # set up sampler
p0 = np.array([theta0 + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]) # initial walkers
pos, prob, state = sampler.run_mcmc(p0, 120) # burn-in run, get new walkers
sampler.reset()

# do proper runs
sampler.run_mcmc(pos, 500)

# plot densities
f3, prm_ax = plt.subplots(3, figsize=(5,5))
parm_title = [r'$r_c$ (m)',r'$\gamma$',r'$\rho_0$ (1000 ev/$m^2$)']
parm_label = [r'$r_c$',r'$\gamma$',r'$\rho_0$']
for i in range(ndim):
    if i == 2:
        prm_ax[i].hist(sampler.flatchain[:,i]*1000, 100, color="k", histtype="stepfilled")
        prm_ax[i].annotate(r'$\sigma = {:.2}$'.format(np.std(sampler.flatchain[:,i]*1000)), xy = (3,400), xycoords = 'data', ha = 'right', va = 'top')
    else:
        prm_ax[i].hist(sampler.flatchain[:,i], 100, color="k", histtype="stepfilled")
        prm_ax[i].annotate(r'$\sigma = {:.2}$'.format(np.std(sampler.flatchain[:,i])), xy = (12,-12), xycoords = 'axes points', ha = 'right', va = 'top')
    prm_ax[i].set_title("{}".format(parm_title[i]))
#prm_ax[2].set_xticklabels(['{:.1e}'.format(t) for t in prm_ax[2].get_xticks()])
plt.tight_layout()
plt.savefig('prm_dstr.png',dpi = 400)

plt.show()

samples = sampler.flatchain

f2, ax2 = plt.subplots(1, figsize = (7,4))
ax2.plot(r, densities, 'o') 
#ax.set_ylim(ax.get_ylim())
for i in range(len(samples)):
    ax2.plot(rplot, (eq.rho(rplot, samples[i][2], samples[i][0], samples[i][1])),'-',color='b',alpha=0.01,lw=.2)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title(fname.split(sep=".")[0])
plt.savefig('newberry_mcmc.png',dpi=400)

#==============================================================================
## perform particle swarm optimisation (MLE)
#rho0 = np.mean(densities[0:5])
#rmax = (r.max())
#rmin = (r.min())
#n_edges = 30
#q = 12
##bin_edges = np.linspace(np.log10(rmin), np.log10(rmax), n_edges) #np.array([r[i] for i in range(0, len(r), q)])
##bin_edges = 10**bin_edges
#bin_edges = np.linspace(rmin, rmax, n_edges) #np.array([r[i] for i in range(0, len(r), q)]) 
##if len(r) % q != 0:
##    bin_edges = np.concatenate((bin_edges, r[-1:(-(len(r) % q)):-1][::-1])) # append obsevations in last bin that werent enough to fill a bin of q 
##n_edges = len(bin_edges)
#const = (rmax, rmin, r, bin_edges, n_edges, rho0)
#
#lb = [50, 1]
#ub = [900, 6]
#bounds = [(low, high) for low, high in zip(lb,ub)] # basinhopping bounds
#
## do particle swarm opti.
##theta0, obj = pso(eq.LLK_rho, lb, ub, args = const, maxiter = 100, swarmsize = 500)
##theta0 = [212.8, 4.4]
##minimizer_kwargs = {"args":const, "bounds":bounds, "method":"L-BFGS-B"}
##annealed = optimize.basinhopping(eq.LLK_rho, theta_guess, minimizer_kwargs = minimizer_kwargs, niter = 500)
##theta0 = annealed.x
## plots
#f, ax = plt.subplots(1, figsize = (7,4))
#
#ax.plot(r, densities, 'o') 
##ax.set_ylim(ax.get_ylim())
#rplot = np.linspace((rmin),(rmax),500)
##ax.plot(rplot, (eq.rho(rplot, rho0, theta0[0], theta0[1])),'-',color='r')
##for be in bin_edges:
##    ax.axvline(be,color='k',linestyle=':')
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.set_title(fname.split(sep=".")[0])
#
#
## MCMC sampling
#
#theta0 = np.array([ 209.87834548,    4.27225937]) # initial walker guess from pswarm
#
#ndim = 2
#nwalkers = 30
#
#sampler = emcee.EnsembleSampler(nwalkers, ndim, eq.LLK_rho, args = [rmax, rmin, r, bin_edges, n_edges, rho0]) # set up sampler
#p0 = np.array([theta0 + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]) # initial walkers
#pos, prob, state = sampler.run_mcmc(p0, 100) # burn-in run, get new walkers
#sampler.reset()
#
## do proper runs
#sampler.run_mcmc(pos, 150)
#
## plot densities
#for i in range(ndim):
#    plt.figure()
#    plt.hist(sampler.flatchain[:,i], 100, color="k", histtype="step")
#    plt.title("Dimension {0:d}".format(i))
#
#plt.show()
#
#samples = sampler.flatchain
#
#f2, ax2 = plt.subplots(1, figsize = (7,4))
#ax2.plot(r, densities, 'o') 
##ax.set_ylim(ax.get_ylim())
#for i in range(200):
#    ax2.plot(rplot, (eq.rho(rplot, rho0, samples[i][0], samples[i][1])),'-',color='b',alpha=0.1,lw=.5)
#ax.set_ylim(1e-4,1e-2)
#ax2.set_xscale('log')
#ax2.set_yscale('log')
#ax.set_title(fname.split(sep=".")[0])



##==============================================================================

print(datetime.now() - start)