# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 16:15:12 2018

@author: enic156
"""

import eq_functions as eq
from matplotlib import pyplot as plt


#rnd.seed(a=1756)
#np.random.seed(1756)


# define parameters
Nt = 1000 # total earthquakes to randomly sample at each iteration
cprime = 1.
pprime = 1.8

locations = eq.sample_location(Nt, cprime, pprime)
plt.hist(locations,color="darkblue",edgecolor="black",bins=100)
