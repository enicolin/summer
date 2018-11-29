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
Tf = 32 # unit time
b = 1.
c = 1.
cprime = 1.
p = 1.1
pprime = 1.8
Mc = 3.
smin = 0.6 # minimum seismicity allowable on an interval so that it doesn't get too small
M0 = 6. # magnitude of initial earthquake
A = 1. # parameter included in law for generating expected aftershocks given main shock magnitude M0
alpha = 2 # parameter included in law for generating expected aftershocks given main shock magnitude M0

prms = pd.Series([Tf,M0,A,alpha,b,c,cprime,p,pprime,Mc,smin],
                 index = ['Tf','M0','A','alpha','b','c','cprime','p','pprime','Mc','smin'])


# generate catalog
catalog_list = []
t0 = 0
gen = 0 # generation
eq.generate_catalog(prms, t0, catalog_list, gen, recursion = True)

#print(datetime.now() - start)

# plot catalog
#eq.plot_catalog(catalog_list, M0, color = 'Generation')
eq.plot_catalog(catalog_list, M0, color = 'Time')

print(datetime.now() - start)