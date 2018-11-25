import numpy as np
import random as rnd
import eq_functions as eq
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1756)
rnd.seed(1756)

# define parameters
Nt = 100
Tf = 12 # unit time
a = np.log10(Nt)
b = 1.
c = 1.
cprime = 1.
p = 1.1
pprime = 1.8
Mc = 3.

#M0 = eq.sample_magnitudes(1,Mc,b) # magnitude of initial earthquake

times = np.linspace(0,Tf,3*Tf+1) # time intervals
dt = times[1]-times[0]# define time increment

# intended column order
cols = ['Average aftershock frequency','Events','Magnitude','Distance','Time']
events_occured = 0 # number of earthquakes generated 
for t in times[:-1]: # for each time interval
    # average seismicity rate on interval [t,t+dt]
    n_avg = eq.average_seismicity(t,t+dt,Tf,a,p,c)
    
    # generate number of events according to a Poisson process
    X = int((eq.sample_poisson(n_avg,1)))
    
    # assign each event a magnitude according to GR
    mgtds = eq.sample_magnitudes(X, Mc, b)
    
    distances = eq.sample_location(X, cprime, pprime)
    
    # generate the times at which each event occurs, according to an exponential distribution.
    # the parameter for the exponential distribution at time interval [t,t+dt] is the expected number
    # of events on this interval according to the Omori law
    times = np.zeros(X)
    inter_times = eq.sample_intereventtimes(n_avg/dt, X)
    if t == 0:
        for i in range(1,X):
            times[i] = inter_times[i] + times[i-1]
        t_start = times[-1] # carry over to next iteration the final event time at current interval
    else:
        for i in range(X):
            if i == 0: # initial time is based off previous iterations final time
                times[i] = t_start + inter_times[i]
            else:
                times[i] = times[i-1] + inter_times[i]
    
    # store results in dataframe
    if t == 0: # initial dataframe, full dataframe constructed via concatenation in subsequent iterations
        # index label for current time interval
        if X != 0:
            interval = [''] * X
            interval[0] = ['Interval: [{:.2f},{:.2f}]'.format(t,t+dt)] # only include interval label on first row
            # create dataframe using dict of objects
            Xcol = ['']*X # only include number of events on first row
            Xcol[0] = X
            n_avgcol = ['']*X
            n_avgcol[0] = n_avg # only include average number of events on first row
            catalog = pd.DataFrame({'Magnitude': mgtds,
                                   'Events':Xcol,
                                   'Average aftershock frequency':n_avgcol,
                                   'Distance':distances,
                                   'Time':times}, index = interval)
            catalog = catalog.reindex(columns = cols)
        else: # formatting for when there are no events during a time interval
            interval = ['Interval: [{:.2f},{:.2f}]'.format(t,t+dt)]
            catalog = pd.DataFrame({'M': ['-'],
                                      'X':[X],
                                      'n_avg':[n_avg],
                                      'Distance':['-'],
                                      'Time':['-']}, index = interval)
            catalog = catalog.reindex(columns = cols)
    else: # join new results to existing catalog
        # index label for current time interval
        if X != 0: 
            interval = [''] * X
            interval[0] = ['Interval: [{:.2f},{:.2f}]'.format(t,t+dt)]# * X
            Xcol = ['']*X
            Xcol[0] = X
            n_avgcol = ['']*X
            n_avgcol[0] = n_avg
            catalog_update = pd.DataFrame({'Magnitude': mgtds,
                                      'Events':Xcol,
                                      'Average aftershock frequency':n_avgcol,
                                      'Distance':distances,
                                      'Time':times}, index = interval)
            catalog_update = catalog_update.reindex(columns = cols)
        else: # formatting for when there are no events during a time interval
            interval = ['Interval: [{:.2f},{:.2f}]'.format(t,t+dt)]
            catalog_update = pd.DataFrame({'Magnitude': ['-'],
                                      'Events':[X],
                                      'Average aftershock frequency':[n_avg],
                                      'Distance':['-'],
                                      'Time':['-']}, index = interval)
            catalog_update = catalog_update.reindex(columns = cols)
        frames = [catalog, catalog_update]
        catalog = pd.concat(frames)

    events_occured += X

#catalog.to_csv('catalog.csv')