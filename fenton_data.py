# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 12:04:32 2019

@author: enic156
"""
import matplotlib.pyplot as plt
import eq_functions as eq
import pandas as pd
from datetime import datetime
import numpy as np
import random


# read in fenton hill data and store event objects in list
f = open("FentonHillExpt2032-MHF-goodlocs.txt",'r')
flines = f.readlines()
flines = flines[0::2]

events = [eq.Event(float(line.split()[11]), '-', float(line.split()[7]), float(line.split()[6]), '-', (float(line.split()[6])**2 + float(line.split()[7])**2)**0.5, '-') for line in flines]

f.close()

#K = [250,500,750,1000,1250,1500,1750,2000,2250]
#times = []
#for k in K:
start = datetime.now()




# reduce events to a random sample of k elements
k = 100
events = random.sample(events, k)

# format events in the pd dataframe format defined by generate_catalog etc. 
catalog = pd.DataFrame({'Magnitude': [2.3] * len(events),
                                   'Events':'-',
                                   'n_avg':'-',
                                   'Time':[0] * len(events),
                                   'Distance':['-'] * len(events),
                                   'x':[event.x for event in events],
                                   'y':[event.y for event in events],
                                   'Generation':[0] * len(events),
                                   'Distance_from_origin': [np.exp(event.distance_from_origin/1000) for event in events]})
cols = ['n_avg','Events','Magnitude','Generation','x','y','Distance','Time','Distance_from_origin']
catalog.reindex(columns = cols)

eq.plot_ED(catalog)
#eq.plot_catalog(catalog, 2, np.array([0,0]), color = 'Generation')
    
#    times.append(datetime.now().timestamp() - start.timestamp())
print(datetime.now().timestamp() - start.timestamp())

#plt.plot(K, times)
#plt.show()