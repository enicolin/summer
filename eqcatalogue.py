import numpy as np
import random as rnd
import eq_functions as eq

# define parameters
a = 3.2
b = 0.97
c = 0.1
p = 1.1

M0 = 5.0 # magnitude of initial earthquake
n0 = eq.GR_M(M0, a, b) # get initial frequency of earthquakes (magnitude M0) using GR

k = n0 * c**p # use initial frequency to know the k parameter
dt = 0.5 # define time increment

n_incr = 20
t = 0 # current time 
for n in range(1,n_incr): # catalogue for n_incr time increments
    t += dt
    n_avg = eq.omori(t, k, c, p) # average freq. of quakes at time t + ndt
    X = np.random.poisson(n_avg)
    # to complete