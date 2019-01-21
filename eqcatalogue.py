import eq_functions as eq
import pandas as pd
import numpy as np
import random as rnd
from datetime import datetime
from pyswarm import pso
import matplotlib.pyplot as plt

start = datetime.now()

np.random.seed(1756)
rnd.seed(1756)

# define parameters
#Nt = 1000
Tf = 100 # forecast period
b = 1.
c = 1.
cprime = 1.
p = 1.1
pprime = 1.8
Mc = 2.
smin = 0.5 # minimum seismicity allowable on an interval so that it doesn't get too small
M0 = 5.7 # magnitude of initial earthquake
A = 1.1 # parameter included in law for generating expected aftershocks given main shock magnitude M0
alpha = 1.4 # parameter included in law for generating expected aftershocks given main shock magnitude M0

t0 = 0 # time of main shock
r0 = np.array([0,0]) # x,y coord. of main shock
gen = 0 # initial generation

## generate catalog and save
#catalog_list = []
#eq.generate_catalog(t0, r0, catalog_list, gen, True,
#                    Tf,M0,A,alpha,b,c,cprime,p,pprime,Mc,smin)

#catalogs = pd.concat(catalog_list)
#catalogs.to_pickle('catalogs.pkl')


# read in catalog and plot
catalogs_raw = pd.read_pickle('catalogs.pkl') # read in .pkl file storing dataframe generated above

# plot catalog
#eq.plot_catalog(catalogs_raw, M0, r0, color = 'Density')
#eq.plot_catalog(catalogs_raw, M0, r0, color = 'Time')

#eq.plot_catalog(catalog, 1, np.array([0,0]), color = 'Generation')
distances, densities = eq.plot_ED(catalogs_raw, plot = False) # get distance, density

# need to be in log space
#distances = np.log10(distances)
#densities = np.log10(densities)

# perform particle swarm optimisation in parameter space on log likelihood
rho0 = np.mean(densities[0:5])
rmax = distances.max()
r = distances
const = (rho0, rmax, r)

lb = [1e-5, 1e-5]
ub = [r.max(), 5]

theta0, llk0 = pso(eq.LLK_rho, lb, ub, args = const, maxiter = 100)

f2, ax2 = plt.subplots(1, figsize = (7,7))
ax2.plot(distances, densities, 'o')
ax2.plot((distances), eq.rho((distances), rho0, theta0[0], theta0[1]),'-',color='r')
ax2.set_xscale('log')
ax2.set_yscale('log')
print(datetime.now() - start)


print(datetime.now() - start)