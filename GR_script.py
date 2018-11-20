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

rnd.seed(a=1756)

# Script to generate a sample of 100 earthquake magnitues using GR law, and
# determine the largest magnitude of these 100 eventsm Mmax. Repeat many times to 
# estimate probability density function of Mmax.

# define parameters
Nt = 100 # total earthquakes to randomly sample at each iteration
a = np.log10(Nt)
b = 0.97
Mc = 2.2

n = int(7e3) # number of times to repeat the process

Mmax = np.empty(n) # initialise array containing largest magnitude event at each iteration
for i in range(n):
    events = eq.sample_magnitudes(Nt, Mc, b) # sample Nt magnitudes based off the distribution according to GR
    
#    # see what kind of distribution quakes are sampled from..
#    plt.figure(1)
#    plt.hist(events, color="darkblue", edgecolor = "black", bins = 100)
#    plt.show()
    
    Mmax[i] = events.max()


plt.figure(1)
plt.hist(Mmax, color="red", edgecolor = "black", bins = 100)
plt.title('Distribution of {} for {} events'.format("$M_{max}$", Nt))
plt.xlabel('$M_{max}$')
plt.ylabel('Occurences')
plt.show()

plt.figure(2)
sns.distplot(Mmax, hist=True, kde=True, bins = 100, color = "black")
plt.title('kernel density estimation (mostly for visualisation right now)')
plt.xlabel('$M_{max}$')
plt.show()