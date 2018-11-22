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
Mc = 2

#M0 = eq.sample_magnitudes(1,Mc,b) # magnitude of initial earthquake


dt = Tf*1e-2 # define time increment

t = 0 # current time
events_occured = 0 # number of earthquakes generated 
#iterations = -1
while events_occured < Nt: # catalogue for n_incr time increments
#    iterations += 1
#    print(iterations)
    # average seismicity rate on interval [t,t+dt]
    n_avg = eq.average_seismicity(t,t+dt,Tf,a,p,c)
    
    # generate number of events according to a Poisson process
    X = int((eq.sample_poisson(n_avg,1)))
    
    # assign each event a magnitude according to GR
    mgtds = eq.sample_magnitudes(X, Mc, b)
    
    # store results in dataframe
    
    cols = ['n_avg','X','M']
    if t == 0: # initial dataframe, full dataframe constructed via concatenation in subsequent iterations
        # index label for current time interval    
        interval = ["Interval: [{},{}]".format(t,t+dt)] * X # length is X - the number of events
        # create dataframe using dict of objects
        catalog = pd.DataFrame({'M': mgtds,
                               'X':[X]*X,
                               'n_avg':[n_avg]*X}, index = interval)
    else: # join new results to existing catalog
        # index label for current time interval    
        interval = ["Interval: [{},{}]".format(t,t+dt)] * X
        catalog_update = pd.DataFrame({'M': mgtds,
                                      'X':[X]*X,
                                      'n_avg':[n_avg]*X}, index = interval)
        frames = [catalog, catalog_update]
        catalog = pd.concat(frames)
    
    if X == 0 and np.abs(n_avg) < 1:
        break
    
    events_occured += X
    t += dt
    
catalog = catalog.reindex(columns = cols) # order columns the way I intend