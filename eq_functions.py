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
    # inverse of F, where F = F(x) = P(X<=x), the probability of having X earthquakes less than magnitude x
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

def omori(t,Tf,c,p,a):
    # Using the Omori aftershock decay law, determine the frequency of aftershocks at a time t after a main shock
    # Tf - forecast period
    
    # determine k proportionality constant by integrating frequency along forecast period and equating to 10^a = total events during forecast period
    k = 10**a * (1-p)/((c+Tf)**(1-p)-(c)**(1-p))
    
    n = k/(c+t)**p

    return n

def poisson_cumul(lmbd,x):
    # returns P(X<=x) where X ~ Poisson(lmbd)
    p = 0
    for k in range(x+1):
        p += lmbd**k/np.math.factorial(k)
    
    p *= np.exp(-lmbd)
    
    return p

def sample_poisson(lmbd,n):
    # sample randomn n numbers from a poisson distribution
    
    # have decided to have possible Poisson numbers in range [0,lambda*k],
    # where k = ceiling(-log10(0.04*lambda) + 2). Scale factor is based off wanting to keep
    # Poisson numbers far enough from the most likely value so that its probability is very low for the interval boundary
    # I don't think a constant scale factor is good enough for low lambda, which is why I used a curve
    # that starts high and quickly trails off to being approximately constant (and the ceiling is needed for integrality anyway)
    
    # largest poisson number depnding on lambda
    if lmbd == 1: # function does not work well as intended for 1, so 
        c = 10
    else:
        c = int(lmbd * np.ceil(-np.log10(0.04*lmbd) + 2))
    
    # generate cumulative probability intervals
    
    intv = np.zeros(c+2) # array with room for all poisson numbers plus an extra zero (for the interval)
    intv[0] = 0 
    for i in range(1,c+2):
        intv[i] = poisson_cumul(lmbd,i-1)
        
    # get n Poisson numbers
    poiss = np.zeros(n)
    assigned = False # flag indicating whether a uniform random number was detectable in the interval
    for k in range(n):
        # generate randomly uniform number and determine which Poisson number it
        # corresponds to
        u = np.random.uniform(0,1)
        for i in range(c+1):
            if u >= intv[i] and u < intv[i+1]: # if the random number is in the enclosed interval, assign it the corresponding Poisson number
                poiss[k] = i
                assigned = True # flag indicating whether a uniform random number was detectable in the interval
                break # leave loop asap
        if not assigned:
            poiss[k] = c # if the number was not found in the interval somehow, assign it the largest poisson number
        
    return poiss
            