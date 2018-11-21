import numpy as np
import random as rnd
import eq_functions as eq
import pandas as pd
import matplotlib.pyplot as plt

# define parameters
Nt = 100
Tf = 15 # unit time
a = np.log10(Nt)
b = 0.97
c = 0.1
p = 1.1
Mc = 2.0

M0 = eq.sample_magnitudes(1,Mc,b) # magnitude of initial earthquake
#n0 = eq.GR_M(0, Tf, c, p, a) # get initial frequency of earthquakes (magnitude M0) using GR

dt = 0.5 # define time increment

t = 0 # current time
events_occured = 0 # number of earthquakes generated 
while events_occured <= Nt: # catalogue for n_incr time increments
    t += dt
    n_avg = eq.omori(0, Tf, c, p, a) # average freq. of quakes at time t + ndt
    X = np.random.poisson(n_avg)
    # to complete