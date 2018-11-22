import numpy as np
import random as rnd
import eq_functions as eq
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1756)
rnd.seed(1756)

# define parameters
Nt = 100
Tf = 15 # unit time
a = np.log10(Nt)
b = 0.97
c = 1
p = 1.1
Mc = 2.0

#M0 = eq.sample_magnitudes(1,Mc,b) # magnitude of initial earthquake


dt = Tf/50 # define time increment

t = 0 # current time
events_occured = 0 # number of earthquakes generated 
while events_occured <= Nt: # catalogue for n_incr time increments
    # average seismicity rate on interval [t,t+dt] is being taken as the mean of n(t) and n(t+dt)
    nt0 = eq.omori(t, Tf, c, p, a)
    nt1 = eq.omori(t+dt, Tf, c, p, a)
    n_avg = np.mean([nt0,nt1])
    
    # generate number of events according to a Poisson process
    X = int((eq.sample_poisson(n_avg,1)))
    
    # assign each event a magnitude according to GR
    mgtds = eq.sample_magnitudes(X, Mc, b)
    
    # store results in dataframe
    if t == 0: # initial dataframe, full dataframe constructed via concatenation in subsequent iterations
        # index label for current time interval    
        interval = ["Interval: [{},{}]".format(t,t+dt)] * X
        catalog = pd.DataFrame({'n_avg':[n_avg]*X,
                               'X':[X]*X,
                               'M': mgtds}, index = interval)
    else: # join new results to existing catalog
        # index label for current time interval    
        interval = ["Interval: [{},{}]".format(t,t+dt)] * X
        catalog_update = pd.DataFrame({'n_avg':[n_avg]*X,
                               'X':[X]*X,
                               'M': mgtds}, index = interval)
        frames = [catalog, catalog_update]
        catalog = pd.concat(frames)
    
    events_occured += X
    t += dt