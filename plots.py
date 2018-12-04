# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:31:04 2018

@author: enic156
"""

import eq_functions as eq
import numpy as np
import random as rnd
import pandas as pd
import matplotlib.pyplot as plt

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

## generate catalog
#catalog_list = []
#t0 = 0 # time of main shock
#r0 = np.array([0,0]) # x,y coord. of main shock
#gen = 0 # initial generation
#eq.generate_catalog(t0, r0, catalog_list, gen, True,
#                    Tf,M0,A,alpha,b,c,cprime,p,pprime,Mc,smin)
#
#catalogs = pd.concat(catalog_list)
#
#catalogs.to_pickle('catalogs.pkl')


######

catalogs_raw = pd.read_pickle('catalogs.pkl') # read in .pkl file storing dataframe generated above

eq.catalog_plots(catalogs_raw)
