# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:08:31 2019

@author: enic156
"""
import eq_functions as eq
import numpy as np
import pandas as pd
import random

k = 2000 # max number of events considered
fname = 'bradys.txt'
f = open(fname, 'r')
flines = f.readlines()
if len(flines) > k:
    # reduce events to a random sample of k elements
    flines = random.sample(flines, k)

# get longitudes and latitudes
lats = [float(event.split()[7]) for event in flines]
longs = [float(event.split()[8]) for event in flines]
latlong = pd.DataFrame({'latitude':lats, 'longitude': longs})
lat_mean = latlong.describe().loc['mean'].loc['latitude']
long_mean = latlong.describe().loc['mean'].loc['longitude']
r = 6.3781e6 # earth radius in m

# store events in list
# NOTES:
# latitude and longitude are converted to rectangular coordinates by
# projecting the sines of the lat. and long. on to the plane tangent to the point of mean lat. and long.
events = [eq.Event(float(event.split()[10]), '-', r*np.sin(np.pi/180*(float(event.split()[8])-long_mean)), r*np.sin(np.pi/180*(float(event.split()[7])-lat_mean)), '-', ((r*np.sin(np.pi/180*(float(event.split()[7])-lat_mean)))**2+(r*np.sin(np.pi/180*(float(event.split()[8])-long_mean)))**2)**0.5, 0) for event in flines]

f.close()

# plot data

# format events in the pd dataframe format defined by generate_catalog etc. 
catalog = pd.DataFrame({'Magnitude': [event.magnitude for event in events],
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
catalog = catalog[catalog.Magnitude <= 6]

print("EVENT POSITIONS AND DENSITY FOR {}".format(fname.split(sep='.')[0]).upper())
eq.plot_catalog(catalog, 2, np.array([0,0]), color = 'Generation')
eq.plot_ED(catalog)