import numpy as np
import random as rnd

def GR_M(M,a,b,Mc):
    # Use the Gutenberg-Richter law to give the number of events of at least magnitude M over a time period
    # (function of M)
    
    N = 10**(a-b*(M-Mc))
    return N

def GR_N(N,a,b,Mc):
    # Given the number of events, use Gutenberg-Richter law to determine the magnitude of the smallest event
    # (function of N)
    
    M = (-1/b)*(np.log10(N)-a-b*Mc)
    return M

def GR_inv(x,Mc,b):
    # inverse of F, where F = F(x) = P(X<=x) - the probability of having X earthquakes less than magnitude x
    # based off Gutenberg-Richter
    # x is a uniformly random number on [0,1]
    
    Finv = Mc - (1/b)*np.log10(1-x)
    return Finv

def sample_magnitudes(n,Mc,b):
    # sample n earthquake events given appropriate parameters based off GR
    # uses the probability integral transform method
    # returns array of length n, whose ith element is the magnitude of the ith event
    
    events = np.empty(n) # initialise 
    for i in range(n):
        xi = rnd.uniform(0,1) # pseudorandom number on [0,1] from a uniform distribution
        events[i] = GR_inv(xi, Mc, b)
        
    return events

def omori(t,k,c,p):
    # Using the Omori aftershock decay law, determine the frequency of aftershocks at a time t after a main shock
    n = k/(c+t)**p

    return n