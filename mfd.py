# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 09:21:42 2018

@author: enic156
"""

import eq_functions as eq
import random as rnd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#rnd.seed(a=1756)
#np.random.seed(1756)

# Script to generate a sample of 100 earthquake magnitues using GR law, and
# determine the largest magnitude of these 100 eventsm Mmax. Repeat many times to 
# estimate probability density function of Mmax.

# define parameters
Nt = 1000 # total earthquakes to randomly sample at each iteration
a = np.log10(Nt)
b = 1.
Mc = 3.

f,ax = plt.subplots(1,1,figsize=(8,8))

for i in range(100):

	events = eq.sample_magnitudes(Nt, Mc, b) # sample Nt magnitudes based off the distribution according to GR

	Nbins = 40
	bin_edges=np.linspace(np.min(events), np.max(events), Nbins+1)
	bin_centers=0.5*(bin_edges[1:]+bin_edges[:-1])
	h, e =np.histogram(events, bins=bin_edges)
	bin_centers,yt=np.array([[m,h] for m, h in zip(bin_centers,h) if h>=1]).T
	N=np.flipud(np.cumsum(np.flipud(yt)))

	ax.plot(bin_centers, N, 'k-', alpha=0.1, lw=0.5)
ax.set_xlabel('M')
ax.set_ylabel('N(>M)')
ax.set_yscale('log')

plt.show()

