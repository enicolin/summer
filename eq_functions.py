import numpy as np
import random as rnd
import decimal as dec

def GR_M(M,a,b,Mc):
    # Use the Gutenberg-Richter law to give the number of events of at least magnitude M over a time period
    # (function of M)
    #
    # Inputs:
    # M -> magnitude
    # a, b -> quantity, slope parameters, respectively
    # Mc -> completeness magnitude
    #
    # Outputs:
    # N -> number of events of magnitude at least M
    
    N = 10**(a-b*(M-Mc))
    return N

def GR_N(N,a,b,Mc):
    # Given the number of events, use Gutenberg-Richter law to determine the magnitude of the smallest event
    # (function of N)
    #
    # Inputs:
    # N -> number of events
    # a, b -> quantity, slope parameters, respectively
    # Mc -> completeness magnitude
    #
    # Outputs:
    # M -> smallest expected magnitude given N events
    
    M = (-1/b)*(np.log10(N)-a-b*Mc)
    return M

def GR_inv(x,Mc,b):
    # inverse of F, where F = F(x) = P(X<=x), the probability of having X earthquakes less than magnitude x in a time period
    # based off Gutenberg-Richter. needed for sampling events according to GR law
    #
    # Inputs:
    # x -> a (uniformly random) number on [0,1]
    # Mc -> completeness magnitude
    # b -> slope parameter
    #
    # Outputs:
    # F_inv -> y such that F(y) = x, where F is defined above
    
    
    Finv = Mc - (1/b)*np.log10(1-x)
    return Finv

def sample_magnitudes(n,Mc,b):
    # sample n earthquake events given appropriate parameters based off GR.
    # uses the probability integral transform method.
    #
    # Inputs:
    # n -> number of events to sample
    # Mc -> completeness magnitude
    # b -> slope parameter
    #
    # Outputs:
    # events -> array of length n, whose ith element is the magnitude of the ith event
    
    events = np.zeros(n) # initialise 
    for i in range(n):
        xi = rnd.uniform(0,1) # pseudorandom number on [0,1] from a uniform distribution
        events[i] = GR_inv(xi, Mc, b)
        
    return events

def omori(t,Tf,c,p,a):
    # Using the Omori aftershock decay law, determine the frequency of aftershocks at a time t after a main shock
    #
    # Inputs:
    # t -> time from main shock
    # Tf - forecast period
    # c, p, a -> c, p, quantity parameters, respectively -- p not equal to 1
    #
    # Outputs:
    # n -> frequency at time t
    
    
    # determine k proportionality constant by integrating frequency along forecast period and equating to 10^a = total events during forecast period
    k = 10**a * (1-p)/((c+Tf)**(1-p)-(c)**(1-p))
    
    n = k/(c+t)**p

    return n

def poisson_cumul(lmbd,x):
    # returns P(X<=x) where X ~ Poisson(lmbd)
    #
    # Inputs:
    # lmbd -> E[X] = Var[X], given X ~ Poisson(lmbd)
    # x -> x such that P(X<=x)
    #
    # Outputs:
    # p -> P(X<=x)
    
    # possibly temporary fix for case large lambda
    lmbd = dec.Decimal(lmbd)
    
    p = 0
    for k in range(x+1):
        p += lmbd**k/np.math.factorial(k)
    
    p *= np.exp(-lmbd)
    
    return p

def sample_poisson(lmbd,n):
    # sample randomn n numbers from a poisson distribution
    #
    # Inputs:
    # lmbd -> E[X] = Var[X], given X ~ Poisson(lmbd)
    # n -> number of events to sample
    #
    # Outputs:
    # poiss -> array of length n containing numbers sampled from a Poisson distribution
    
    
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

def average_seismicity(t_low,t_upp,Tf,a,p,c):
    # take the mean of n as given by the Omori law on interval [t_low,t_upp] using the definite integral
    #
    # Inputs:
    # t_low -> lower time limit
    # t_upp -> upper time limit
    # Tf -> forecast period
    # a, p, c -> parameters in Omori law -- p not equal to 1
    #
    # Outputs:
    # n_avg -> 1/(t_upp-t_low) * integral from t_low to t_upp of n(t) dt
    
    k = 10**a * (1-p)/((c+Tf)**(1-p)-(c)**(1-p))
    
    n_avg = 1/(t_upp-t_low) * k * ((c+t_upp)**(1-p)/(1-p) -(c+t_low)**(1-p)/(1-p))
    
    return n_avg
    