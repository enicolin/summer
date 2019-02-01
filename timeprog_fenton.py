# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:37:38 2019

@author: enic156
"""

import eq_functions as eq
import pandas as pd
from datetime import datetime
import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(1756)
random.seed(1756)

# read in fenton hill data and store event objects in list
f = open("FentonHillExpt2032-MHF-goodlocs.txt",'r')
flines = f.readlines()
flines = flines[0::2]

time_init = datetime(1983,12,6,hour=22,minute=4,second=36)
events = [eq.Event(float(line.split()[11]),\
                   (datetime(int('19'+line.split()[1][:2]), int(line.split()[1][2:]),int(line.split()[2][:2]),hour=int(line.split()[2][2:]),minute=int(line.split()[3][:2]),second=int(line.split()[3][2:]))-time_init).total_seconds()\
                   , float(line.split()[6]), float(line.split()[5]), '-', 0, '-') for line in flines]

f.close()

start = datetime.now()

# reduce events to a random sample of k elements
#k = 650
#events = random.sample(events, k)

# format events in the pd dataframe format defined by generate_catalog etc. 
catalog = pd.DataFrame({'Magnitude': [2.3] * len(events),
                                   'Events':'-',
                                   'n_avg':'-',
                                   'Time':[event.time for event in events],
                                   'Distance':['-'] * len(events),
                                   'x':[event.x for event in events],
                                   'y':[event.y for event in events],
                                   'Generation':[0] * len(events),
                                   'Distance_from_origin': [event.distance_from_origin for event in events]})
cols = ['n_avg','Events','Magnitude','Generation','x','y','Distance','Time','Distance_from_origin']
catalog = catalog.reindex(columns = cols)

r0 = np.array([np.mean([event.x for event in events]), np.mean([event.y for event in events])])#np.array([3557.418383, -324.384367])
catalog['x'] = r0[0] - catalog.x # shift so that main shock position is (0,0)
catalog['y'] = r0[1]-50 - catalog.y
catalog['Distance_from_origin'] = (catalog.x**2 + catalog.y**2)**0.5

#for i in range(1000,3886):
eq.plot_catalog(catalog, 1, np.array([0,0]), color = 'Density')#, savepath = 'fentonprog_frames/{}.png'.format(i), saveplot = True)

print((datetime.now()-start).total_seconds())