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

# define parameters a, b
a = 2
b = 0.97

Nt = 100 # total earthquakes to randomly sample at each iteration
N = 750 # number of times to repeat the process
Mmin = eq.GR_N(Nt, a, b) # given Nt, the smallest magnitude quake we expect to see given G-R

Mmax = np.empty(N) # initialise array containing largest magnitude event at each iteration
for i in range(N):
    events = eq.sample_events(Nt) # array of length Nt containing elements uniformly random between 0 and Nt
    
    # determine corresponding magnitudes for each event using G-R
    magnitudes = np.empty(Nt)
    for k in range(len(events)):
        magnitudes[k] = eq.GR_N(events[k], a, b)
        
    Mmax[i] = magnitudes.max()

plt.figure(1)
plt.hist(Mmax, color="red", edgecolor = "black", bins = 100)
plt.title('Distribution of $M_{max}$')
plt.xlabel('$M_{max}$')
plt.ylabel('Occurences')
plt.show()

#plt.figure(2)
#sns.distplot(Mmax, hist=True, kde=True, bins = 100, color = "black")
#plt.show()