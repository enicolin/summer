import numpy as np
import random as rnd
from math import log
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

class Event:
    '''Earthquake class'''
    
    def __init__(self, magnitude, time, x, y, gen):
        self.magnitude = magnitude
        self.time = time
        self.x = x
        self.y = y
        self.generation = gen
    
    def __repr__(self):
        '''Returns string representation of Event'''
        return("{}({},{},{},{},{})".format(self.__class__.__name__,
               self.magnitude,
               self.time,
               self.x,
               self.y,
               self.generation))

def GR_M(M,a,b,Mc):
    """
    Use the Gutenberg-Richter law to give the number of events of at least magnitude M over a time period
    (function of M)
    
    Inputs:
    M -> magnitude
    a, b -> quantity, slope parameters, respectively
    Mc -> completeness magnitude
    
    Outputs:
    N -> number of events of magnitude at least M
    """
    
    N = 10**(a-b*(M-Mc))
    return N

def GR_N(N,a,b,Mc):
    """
    Given the number of events, use Gutenberg-Richter law to determine the magnitude of the smallest event
    (function of N)
    Inputs:
    N -> number of events
    a, b -> quantity, slope parameters, respectively
    Mc -> completeness magnitude
    
    Outputs:
    M -> smallest expected magnitude given N events
    """
    
    M = (-1/b)*(np.log10(N)-a-b*Mc)
    return M

def GR_inv(u,Mc,b):
    """
    inverse of F, where F = F(x) = P(X<=x), the probability of having X earthquakes less than magnitude x in a time period
    based off Gutenberg-Richter. needed for sampling events according to GR law
    Inputs:
    u -> a (uniformly random) number on [0,1]
    Mc -> completeness magnitude
    b -> slope parameter
    
    Outputs:
    x -> x such that F(x) = u, where F is defined above
    """
    
    x = Mc - (1/b)*np.log10(1-u)
    return x

def sample_magnitudes(n,Mc,b):
    """
    sample n earthquake events given appropriate parameters based off GR.
    uses the probability integral transform method.
    Inputs:
    n -> number of events to sample
    Mc -> completeness magnitude
    b -> slope parameter
    
    Outputs:
    events -> array of length n, whose ith element is the magnitude of the ith event
    """
    
    events = np.zeros(n) # initialise 
    for i in range(n):
        ui = rnd.uniform(0,1) # pseudorandom number on [0,1] from a uniform distribution
        events[i] = GR_inv(ui, Mc, b)
        
    return events

def bath_inv(u,M0,Mc,b):
    """
    inverse of F, where F = F(x) = P(X<=x), the probability of having X earthquakes less than magnitude x in a time period
    based off Gutenberg-Richter. needed for sampling events according to GR law
    Inputs:
    u -> a (uniformly random) number on [0,1]
    M0 -> magnitude of the main shock
    Mc -> completeness magnitude
    b -> slope parameter
    
    Outputs:
    x -> x such that F(x) = u, where F is defined above
    """
    
    dm = 1. # difference between main shock and greatest aftershock according to Båth
    k = 1/(1-10**(-b*(M0-dm-Mc)))
    
    x = Mc - (1/b)*np.log10(1-u/k)
    return x

def sample_magnitudes_bath(n,M0,Mc,b):
    """
    sample n earthquake events given appropriate parameters based off GR.
    uses the probability integral transform method.
    Inputs:
    n -> number of events to sample
    M0 -> magnitude of the main shock
    Mc -> completeness magnitude
    b -> slope parameter
    
    Outputs:
    events -> array of length n, whose ith element is the magnitude of the ith event
    """
    
    events = np.zeros(n) # initialise 
    for i in range(n):
        ui = rnd.uniform(0,1) # pseudorandom number on [0,1] from a uniform distribution
        events[i] = bath_inv(ui, M0, Mc, b)
        
    return events



def omori(t,Tf,c,p,a):
    """
    Using the Omori aftershock decay law, determine the frequency of aftershocks at a time t after a main shock
    Inputs:
    t -> time from main shock
    Tf - forecast period
    c, p, a -> c, p, quantity parameters, respectively -- p not equal to 1
    
    Outputs:
    n -> frequency at time t
    """
    
    # determine k proportionality constant by integrating frequency along forecast period and equating to 10^a = total events during forecast period
    A = 1 - p
    B = (c+Tf)**(1-p)
    C = c**(1-p)
    k = 10**a * A/(B-C)
    
    n = k/(c+t)**p

    return n

def omori_spatial(r,rmax,c,p,a):
    """
    Spatial Omori law
    """
    
    k = 10**a * (1-p)/( (c+rmax)**(1-p) - (c)**(1-p) )
    
    n = k/(c+r)**p
    return n

def poisson(x,lmbd):
    """
    Given X ~ Poisson(lmbd), computes P(X = x)
    Rephrased using logs to handle bigger lambda
    """
    # Inputs:
    # x -> integer
    # lmbd -> expected value
    #
    # Outputs:
    # p -> P(X = x)
    
    p = np.exp(x*log(lmbd) - log(np.math.factorial(x))-lmbd)
    return p

def poisson_cumul(lmbd,x):
    """
    returns P(X<=x) where X ~ Poisson(lmbd)
    Inputs:
    lmbd -> E[X] = Var[X], given X ~ Poisson(lmbd)
    x -> x such that P(X<=x)

    Outputs:
    p -> P(X<=x)
    """
    
    p = 0
    for k in range(x+1):
        p += poisson(k, lmbd)
    
    return p

def sample_poisson(lmbd,n):
    """
    sample randomn n numbers from a poisson distribution
    Inputs:
    lmbd -> E[X] = Var[X], given X ~ Poisson(lmbd)
    n -> number of events to sample

    Outputs:
    poiss -> array of length n containing numbers sampled from a Poisson distribution
    """
    
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
    """
    Get the expected number of events as given by the Omori law on interval [t_low,t_upp] using the definite integral
    Inputs:
    t_low -> lower time limit
    t_upp -> upper time limit
    Tf -> forecast period
    a, p, c -> parameters in Omori law -- p not equal to 1

    Outputs:
    n_avg -> 1/(t_upp-t_low) * integral from t_low to t_upp of n(t) dt
    """
    
    k = 10**a * (1-p)/((c+Tf)**(1-p)-(c)**(1-p))
    
    n_avg = k * ((c+t_upp)**(1-p)/(1-p) -(c+t_low)**(1-p)/(1-p)) #* 1/(t_upp-t_low)
    
    return n_avg

def omori_spatial_inverse(u,p,c):
    """
    Inverse of the cdf wihch gives the probability of having an aftershock at radius r or less, according to spatial Omori
    Inputs:
    u -> a number in [0,1]
    p, c -> p prime, c prime parameters

    Outputs:
    x -> the number x such that N(x) = u
    """
    
    x = (c**(1-p) - u*c**(1-p))**(1/(1-p)) - c
    return x
    
def sample_location(n,c,p):
    """
    Generate distances from main event according to spatial Omori law
    Inputs:
    n -> number of events
    c, p -> c prime and p prime parameters
    
    Outputs:
    locations -> an array of length n whose ith element is the distance of the ith event
    """
    
    locations = np.zeros(n) # initialise 
    for i in range(n):
        ui = rnd.uniform(0,0.95) # pseudorandom number on [0,1] from a uniform distribution
        locations[i] = omori_spatial_inverse(ui,p,c)
        
    return locations
    
def interevent_inverse(u,lmbd):
    """
    The inverse of the cdf of the exponential distribution, for sampling random interevent times
    Inputs:
    u -> a number in [0,1]
    lmbd -> the exponential distribution parameter (average number of events on interval [t,t+t] in our case)
    
    Outputs:
    x -> the number x such that F(x) = u
    """
    x = -(1/lmbd)*np.log(1-u)
    
    return x

def sample_intereventtimes(lmbd,n):
    """
    Generate an array of n interevent times for parameter lmbd

    Inputs:
    lmbd -> exponential distribution parameter (average number of events on interval [t,t+t] in our case)
    n -> the number of interevent times we want to sample
    
    Outputs:
    times -> array of length n whose ith element is the ith interevent time
    """
    times = np.zeros(n) # initialise 
    for i in range(n):
        ui = rnd.uniform(0,1) # pseudorandom number on [0,1] from a uniform distribution
        times[i] = interevent_inverse(ui,lmbd)
        
    return times

def generate_events(n_avg, t, dt, r, M0, Mc, b, cprime, pprime, gen, recursion):
    """
    Generate list of Event objects based off seismicity n_avg and other parameters
    Inputs:
    n_avg -> seismicity
    Mc -> Completeness magnitude
    b, cprime, pprime -> the usual parameters
    
    Outputs:
    events -> array whose length is a number sampled from a Poisson distribution of parameter n_avg
    """
    
    # generate number of events according to a Poisson process
    X = int((sample_poisson(n_avg,1)))
    
    # assign each event a magnitude according to GR
    if recursion: # if recursive, sample from a modified GR distribution where the largest value possible is M0 - dm
        mgtds = sample_magnitudes_bath(X, M0, Mc, b)
    else:
        mgtds = sample_magnitudes(X, Mc, b)
    
    # assign distances according to spatial Omori (with random azimuth angle)
    distances = sample_location(X, cprime, pprime)
    thetas = np.random.uniform(0, 2*np.pi, X)
    
    x = r[0] + distances * np.cos(thetas)
    y = r[1] + distances * np.sin(thetas)
    
    # generate the times at which each event occurs - uniform random number on [t,t+dt]
    times = np.random.uniform(t, t+dt, X)
    times = np.sort(times)
    
    # this is faster than [Event(mgtds[i], times[i], distances[i], thetas[i], gen) for i in range(X)]
    events = [] 
    for i in range(X):
        eventi = Event(mgtds[i], times[i], x[i], y[i], gen)
        events.append(eventi)
    
    return events
        

def generate_catalog(t0, r0, catalog_list, gen, recursion,
                     Tf,M0,A,alpha,b,c,cprime,p,pprime,Mc,smin):
    """
    Generate a synthetic aftershock catalog based off input parameters
    Recursively produces aftershocks for aftershocks
    
     Inputs:
     Tf -> forecast period
     M0 -> mainshock magnitude
     A -> productivity parameter
     alpha -> productivity parameter
     b -> slope parameter
     c -> from Omori
     cprime -> from spatial Omori
     p -> from Omori
     pprime -> from spatial Omori
     Mc -> completeness magnitude
     smin -> seismicity at each time interval
     t0 -> initial time (0)
     r0 -> initial position np.array([x,y])
     catalog_list -> empty list to be populated with generated aftershock catalogs
     gen -> variable to keep track of aftershock generation
    """
    
    # define parameters as local variables so that Series doesn't have to be accessed multiple times
#    A = prms['A']
#    alpha = prms['alpha']
#    M0 = prms['M0']
#    Mc = prms['Mc']
    Nt = A*np.exp(alpha*(M0 - Mc))
#    Tf = prms['Tf']
    a = np.log10(Nt)
#    b = prms['b']
#    c = prms['c']
#    cprime = prms['cprime']
#    p = prms['p']
#    pprime = prms['pprime']
#    smin = prms['smin'] # minimum seismicity allowable on an interval so that it doesn't get too small
    k = 10**a * (1-p)/((c+Tf)**(1-p) - c**(1-p)) # k from Omori -needed for adaptive time increment
    
    # intended column order
    cols = ['n_avg','Events','Magnitude','Generation','x','y','Time']
    events_occurred = 0 # number of earthquakes generated 
    t = t0
#    r = r0
    while t < Tf: # iterate until reached end of forecast period
        dt = (-1/k * smin*(p-1) + (c+t)**(1-p))**(-1/(p-1)) - c - t # update time increment - set up so that seismicity is equal to smin at each interval
        if t + dt > Tf: # if time increment goes over forecast period
            dt = Tf - t
        # average seismicity rate on interval [t,t+dt]
        n_avg = average_seismicity(t,t+dt,Tf,a,p,c)
        
        # generate events - list of Event objects
        events = generate_events(n_avg, t, dt, r0, M0, Mc, b, cprime, pprime, gen, recursion)
        X = len(events)

        # store results in dataframe
        if X != 0:
            interval = ['Interval: [{:.3f},{:.3f}]'.format(t,t+dt)] * X 
            # create dataframe using dict of objects
            Xcol = [''] * X # only include number of events on first row
            Xcol[0] = X
            n_avgcol = [''] * X
            n_avgcol[0] = n_avg # only include average number of events on first row
            catalog_update = pd.DataFrame({'Magnitude': [event.magnitude for event in events],
                                   'Events':Xcol,
                                   'n_avg':n_avgcol,
                                   'Time':[event.time for event in events],
                                   'x':[event.x for event in events],
                                   'y':[event.y for event in events],
                                   'Generation':[event.generation for event in events]}, index = interval)
            catalog_update = catalog_update.reindex(columns = cols)
        else: # formatting for when there are no events during a time interval
            interval = ['Interval: [{:.3f},{:.3f}]'.format(t,t+dt)]
            catalog_update = pd.DataFrame({'Magnitude': [0],
                                      'Events':[0],
                                      'n_avg':[n_avg],
                                      'Time':['-'],
                                      'x':['-'],
                                      'y':['-'],
                                      'Generation':['-']}, index = interval)
            catalog_update = catalog_update.reindex(columns = cols)
        if t == t0:
            catalog = catalog_update
        else:
            frames = [catalog, catalog_update]
            catalog = pd.concat(frames)
    
        events_occurred += X
        t += dt
        
    if recursion:
        parent_shocks = catalog[catalog.Magnitude > Mc] # get susbet of shocks that are able to create aftershocks
        # base case
        if parent_shocks.empty:
            if not catalog[catalog.Magnitude != 0].empty: # only append to catalog list if current catalog contains events > Mc
                catalog_list.append(catalog[catalog.Magnitude > Mc])
            return
        else:
            if not catalog[catalog.Magnitude != 0].empty: # only append to catalog list if current catalog contains events > Mc
                catalog_list.append(catalog[catalog.Magnitude > Mc])
#            prms_child = prms.copy() # create copy of parameters to be modified for next generation of shocks
            for i in range(np.shape(parent_shocks)[0]):
#                prms_child.M0 = parent_shocks.iat[i,2] # main shock
                generate_catalog(parent_shocks.iat[i,6],
                                 np.array([parent_shocks.iat[i,4], parent_shocks.iat[i,5]]),
                                 catalog_list, gen+1, recursion,
                                 Tf,parent_shocks.iat[i,2],A,alpha,b,c,cprime,p,pprime,Mc,smin)
    else:
        catalog_list.append(catalog)
        return

def plot_catalog(catalog_list, M0, r0, color = 'Time'):
    """
     Plots generated synthetic catalog from generate_catalog
     Inputs:
     catalog_list -> output list of pandas DataFrames from generate_catalog
     M0 -> main shock magnitude
     color -> color scheme of events
               'Time' - default, colours events by time of occurrence
               'Generation' - colours by aftershock generation
    """
    fig= plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    ax = fig.add_subplot(111)
    
    total_events = 0 # count how many events took place

    catalogs = pd.concat(catalog_list) # concatenate list of catalogs into one large dataframe     
    catalogs = catalogs[catalogs.Magnitude != 0] # extract rows where events took place
    
    if color == 'Time': # formatting for time option

        x = catalogs['x']#dist * np.cos(theta)
        y = catalogs['y']#dist * np.sin(theta)

        times = catalogs['Time']
        times = np.array(times, dtype = np.float)
        
#        plt.hist(times, bins = 50) # to see if times obey Omori law
#        plt.show()
        
        magnitudes = catalogs['Magnitude']
        magnitudes = np.array(magnitudes, dtype = np.float)
        total_events += len(magnitudes)
        
        c = times
        cmap = 'Spectral'
        # update range for color bar if needed
        cmax = times.max()
        plot = plt.scatter(x, y,
                   c = c,
                   s = 0.05*10**magnitudes, # large events displayed much bigger than smaller ones
                   cmap = cmap,
                   alpha = 0.75)
        plt.clim(0, cmax)
        
        cax = fig.add_axes([0.91, 0.1, 0.075, 10 * 0.08])
        cbar = plt.colorbar(plot, orientation='vertical', cax=cax)
        cbar.set_label(color)
    elif color == 'Generation':
        colors = ['#ffa220', '#ddff20', '#20fffb', '#866be0', '#d83a77'] # haven't seen more than 4 generations, so up to 5 colours should be fine hopefully
        # filter events by generation - in order to provide legend for aftershock generation
        catalogs_bygen = [] # store them in a list
        for i in range(catalogs['Generation'].max()+1):
            catalogs_bygen.append(catalogs[catalogs.Generation == i])
        
        for catalog, event_color in zip(catalogs_bygen, colors):
            # get coordinates
            x = catalogs['x']#dist * np.cos(theta)
            y = catalogs['y']#dist * np.sin(theta)
            
            # get magnitudes for size
            magnitudes = catalog['Magnitude']
            magnitudes = np.array(magnitudes, dtype = np.float)
            total_events += len(magnitudes)
            
            c = np.array(catalog.Generation)[0] # colour is generation
            plt.scatter(x, y,
                       c = event_color,
                       s = 0.05*10**magnitudes, # large events displayed much bigger than smaller ones
                       label = c,
                       cmap = 'Set1',
                       alpha = 0.75)
        lgnd = plt.legend(loc="best", scatterpoints=1, fontsize=18)
        for lgndhndl in lgnd.legendHandles:
            lgndhndl._sizes = [50]
    
    # plot the (initial/parent of all parents) main shock
    ax.scatter(r0[0], r0[1],
           c = '#21ff60',
           alpha = 1,
           marker = "x")
    
    ax.set_title('Generated Events ({})'.format(total_events))
    ax.set_ylabel('y position')
    ax.set_xlabel('x position')
    
    # formatting choices depending on whether viewing by aftershock generation/by time
    if color == 'Generation':
#        ax.set_facecolor('#1b2330')
        pass
    else:
        ax.grid(True)

    plt.show()

def frequency_by_interval(x, nbins):
    """
    For a given array of x values, determine the frequency of elements within nbins equally spaced bins partitioning x
    Returns (x,y) coords where x -> bin center, y -> frequency of x 
    """
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    centers = 0.5*(edges[1:]+edges[:-1])
    frequencies = np.array([sum(1 for i in x if (i >= edges[j] and i <= edges[j+1])) for j in range(nbins)])
    
    return centers, frequencies
    
    

def catalog_plots(catalog_pkl):
    catalogs = catalog_pkl[catalog_pkl.Magnitude != 0] # filter out for non-zero magnitude events
    catalogs = catalogs.sort_values(by = ['Time']) # sort by ascending order by time
    
    catalogs = catalogs.loc[:,['Magnitude','Time']]
    
    time = np.array(catalogs.Time, dtype = float)
    magnitude = np.array(catalogs.Magnitude)
    
    dt = np.array([np.abs(i-j) for i in time for j in time if i != j]).min()
    dt = 0.95 * dt
    
    edges = np.concatenate((np.array([0]),time + dt))
    rates = [1/(edges[i]-edges[i-1]) for i in range(1,len(time)+1)]
    
    f, (ax1, ax2) = plt.subplots(2, figsize=(10,5))
    
    plt.sca(ax1)
    ax1.plot(time, rates, color = 'orange')
    ax1.set_yscale('log')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Fequency of events per unit time')
    ax1.set_title('Seismicity Rate')
    plt.xlim([0, time.max()])
    plt.tight_layout()
    
    plt.sca(ax2)
    markerline, stemlines, baseline = ax2.stem(time, magnitude)
    ax2.set_yscale('log')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Event magnitude with time')
    plt.xlim([0, time.max()])
    
    ax3 = ax2.twinx()
    plt.sca(ax3)
    plt.grid(False)
    nbins = int(time.max()/3.5)
    bin_centers, frequencies = frequency_by_interval(time, nbins)
    sigma = 1
    bin_gauss = gaussian_filter1d(bin_centers, sigma)
    freq_gauss = gaussian_filter1d(frequencies, sigma)
#    ax3.scatter(bin_centers, frequencies, color = 'black')
    ax3.plot(bin_gauss, freq_gauss, color = 'black')
    ax3.set_ylabel('Smoothed event frequency')
    plt.xlim([0, time.max()])
    
    
    plt.show()

