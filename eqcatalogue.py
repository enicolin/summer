import numpy as np
import random as rnd
import eq_functions as eq
import pandas as pd

np.random.seed(1756)
rnd.seed(1756)

# define parameters
Nt = 1000
Tf = 32 # unit time
a = np.log10(Nt)
b = 1.
c = 1.
cprime = 1.
p = 1.1
pprime = 1.8
Mc = 3.
smin = 0.7 # minimum seismicity allowable on an interval so that it doesn't get too small
k = 10**a * (1-p)/((c+Tf)**(1-p) - c**(1-p)) # k from Omori -needed for adaptive time increment

prms = pd.Series([Nt,Tf,b,c,cprime,p,pprime,Mc,smin],
                 index = ['Nt','Tf','b','c','cprime','p','pprime','Mc','smin'])

#M0 = eq.sample_magnitudes(1,Mc,b) # magnitude of initial earthquake

# generate catalog
catalog = eq.generate_catalog(prms)

# plot catalog
eq.plot_catalog(catalog)

#catalog.to_csv('catalog.csv')