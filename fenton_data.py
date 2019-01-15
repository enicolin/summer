# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 12:04:32 2019

@author: enic156
"""
import eq_functions as eq
import pandas as pd
from datetime import datetime
import numpy as np
import random

np.random.seed(1756)
random.seed(1756)

# read in fenton hill data and store event objects in list
f = open("FentonHillExpt2032-MHF-goodlocs.txt",'r')
flines = f.readlines()
flines = flines[0::2]

events = [eq.Event(float(line.split()[11]), '-', float(line.split()[7]), float(line.split()[6]), '-', 0, '-') for line in flines]

f.close()

start = datetime.now()

# reduce events to a random sample of k elements
k = 1000
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
                                   'Distance_from_origin': [event.distance_from_origin for event in events]})
cols = ['n_avg','Events','Magnitude','Generation','x','y','Distance','Time','Distance_from_origin']
catalog = catalog.reindex(columns = cols)

r0 = np.array([3557.418383, -324.384367])
catalog['x'] = r0[0] - catalog.x # shift so that main shock position is (0,0)
catalog['y'] = r0[1] - catalog.y
catalog['Distance_from_origin'] = (catalog.x**2 + catalog.y**2)**0.5

eq.plot_catalog(catalog, 2, np.array([0,0]), color = 'Generation')
eq.plot_ED(catalog)
    
print(datetime.now().timestamp() - start.timestamp())
