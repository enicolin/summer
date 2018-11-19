import numpy as np
import random as rnd

def GR_M(M,a,b):
    # Use the Gutenberg-Richter law to give the number of events of at least magnitude M over a time period
    # (function of M)
    
    N = 10**(a-b*M)
    return N

def GR_N(N,a,b):
    # Given the number of events, use Gutenberg-Richter law to determine the magnitude of the smallest event
    # (function of N)
    
    M = (1/b)*(a-np.log10(N))
    return M

def sample_events(Nt):
    # create an array of length Nt with entries in the range [0,Nt]
    events = np.empty(Nt)
    for i in range(Nt):
        events[i] = rnd.uniform(0.,float(Nt))
        
    return events

def omori(t,k,c,p):
    # Using the Omori aftershock decay law, determine the frequency of aftershocks at a time t after a main shock
    n = k/(c+t)**p

    return n