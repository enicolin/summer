import eq_functions as eq
import pandas as pd
import numpy as np
import random as rnd
from datetime import datetime

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
smin = 0.6 # minimum seismicity allowable on an interval so that it doesn't get too small
M0 = 6. # magnitude of initial earthquake
A = 1.1 # parameter included in law for generating expected aftershocks given main shock magnitude M0
alpha = 1.3 # parameter included in law for generating expected aftershocks given main shock magnitude M0

t0 = 0 # time of main shock
r0 = np.array([0,0]) # x,y coord. of main shock
gen = 0 # initial generation

## generate catalog and save
#catalog_list = []
#eq.generate_catalog(t0, r0, catalog_list, gen, True,
#                    Tf,M0,A,alpha,b,c,cprime,p,pprime,Mc,smin)
#
#catalogs = pd.concat(catalog_list)
#catalogs.to_pickle('catalogs.pkl')


# read in catalog and plot
catalogs_raw = pd.read_pickle('catalogs.pkl') # read in .pkl file storing dataframe generated above

# plot catalog
#eq.plot_catalog(catalogs_raw, M0, r0, color = 'Density')
eq.plot_catalog(catalogs_raw, M0, r0, color = 'Generation')


print(datetime.now() - start)