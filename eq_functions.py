import numpy as np
import random as rnd
from math import log
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as special
from scipy import integrate
from scipy.ndimage.filters import gaussian_filter1d
import mpmath as mp
#from itertools import compress

class Event:
    '''Earthquake class'''
    
    def __init__(self, magnitude, time, x, y, dist, dist_from_origin, gen):
        self.magnitude = magnitude
        self.time = time
        self.x = x
        self.y = y
        self.distance = dist
        self.distance_from_origin = dist_from_origin
        self.generation = gen
    
    def __repr__(self):
        '''Returns string representation of Event'''
        return("{}(mgn = {}, time = {}, x = {}, y = {}, dst = {}, dst_frn_orgn = {}, gen = {})".format(self.__class__.__name__,
               self.magnitude,
               self.time,
               self.x,
               self.y,
               self.distance, # distance from parent shock, not main shock necessarily
               self.distance_from_origin,
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
        ui = rnd.uniform(0,1) # pseudorandom number on [0,1] from a uniform distribution
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
    
    # assign distances from parent shock according to spatial Omori (with random azimuth angle)
    distances = sample_location(X, cprime, pprime)
    thetas = np.random.uniform(0, 2*np.pi, X)
    
    x = r[0] + distances * np.cos(thetas)
    y = r[1] + distances * np.sin(thetas)
    
    # generate the times at which each event occurs - uniform random number on [t,t+dt]
    times = np.random.uniform(t, t+dt, X)
    times = np.sort(times)
    
    dist_to_origin = (x**2 + y**2)**0.5
    
    events = [Event(mgtds[i], times[i], x[i], y[i], distances[i], dist_to_origin[i], gen) for i in range(X)]
    
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
    recursion -> Boolean
    """
    
    # derived parameters
    Nt = A*np.exp(alpha*(M0 - Mc))
    a = np.log10(Nt)
    k = 10**a * (1-p)/((c+Tf)**(1-p) - c**(1-p)) # k from Omori -needed for adaptive time increment
    
    # intended column order
    cols = ['n_avg','Events','Magnitude','Generation','x','y','Distance','Time','Distance_from_origin']
    events_occurred = 0 # number of earthquakes generated 
    t = t0
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
        if X is not 0:
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
                                   'Distance':[event.distance for event in events],
                                   'x':[event.x for event in events],
                                   'y':[event.y for event in events],
                                   'Generation':[event.generation for event in events],
                                   'Distance_from_origin': [event.distance_from_origin for event in events]}, index = interval)
            catalog_update = catalog_update.reindex(columns = cols)
        else: # formatting for when there are no events during a time interval
            interval = ['Interval: [{:.3f},{:.3f}]'.format(t,t+dt)]
            catalog_update = pd.DataFrame({'Magnitude': [0],
                                      'Events':[0],
                                      'n_avg':[n_avg],
                                      'Time':['-'],
                                      'Distance':['-'],
                                      'x':['-'],
                                      'y':['-'],
                                      'Generation':['-'],
                                      'Distance_from_origin':['-']}, index = interval)
            catalog_update = catalog_update.reindex(columns = cols)
        if t is t0:
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
            catalog_list.append(catalog)
            return
        else:
            catalog_list.append(catalog)
            for i in range(np.shape(parent_shocks)[0]):
                r_parent = np.array([parent_shocks.iat[i,4], parent_shocks.iat[i,5]]) # parent shock position (x,y)
                generate_catalog(parent_shocks.iat[i,7],
                                 r_parent,
                                 catalog_list, gen+1, recursion,
                                 Tf,parent_shocks.iat[i,2],A,alpha,b,c,cprime,p,pprime,Mc,smin)
    else:
        catalog_list.append(catalog)
        return

def plot_catalog(catalogs_raw, M0, r0, color = 'Time'):
    """
     Plots generated synthetic catalog from generate_catalog
     Inputs:
     catalog_list -> concatenated output list of pandas DataFrames from generate_catalog
     M0 -> main shock magnitude
     color -> color scheme of events
               'Time' - default, colours events by time of occurrence
               'Generation' - colours by aftershock generation
    """
    fig= plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    ax = fig.add_subplot(111)
  
    catalogs = catalogs_raw[catalogs_raw.Magnitude != 0] # extract rows where events took place
    total_events = len(catalogs) # count how many events took place
    
    if color == 'Time': # formatting for time option

        x = catalogs['x']
        y = catalogs['y']

        times = catalogs['Time']
        times = np.array(times, dtype = np.float)
        
        magnitudes = catalogs['Magnitude']
        magnitudes = np.array(magnitudes, dtype = np.float)
        
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

    elif color == 'Density':
        # display for event density, based off kNN
        
        x = np.array(catalogs['x'])
        y = np.array(catalogs['y'])
        
        # need a list of vectors for position
        npoints = 100
        k = 2
        positions = [np.array(([xi],[yi])) for xi, yi in zip(x,y)]
        xgrid = np.linspace(x.min(), x.max(), npoints)
        ygrid = np.linspace(y.min(), y.max(), npoints)
        points = [np.array(([xi],[yi])) for yi in ygrid for xi in xgrid] # points at which to sample kNN
        xm, ym = np.meshgrid(xgrid, ygrid)
        density = np.zeros(np.shape(xm))
        p = 0
        for i in range(len(xgrid)):
            for j in range(len(ygrid)):
                density[i][j] = k/(2 * total_events * kNN_measure(positions, points[p], k, dim = 2))
                p += 1
        plot = plt.contourf(xgrid, ygrid, density, 30, cmap = 'plasma')
        
        cax = fig.add_axes([0.91, 0.1, 0.075, 10 * 0.08])
        cbar = plt.colorbar(plot, orientation='vertical', cax=cax)
        cbar.set_label('Event density')
        
    # plot the (initial/parent of all parents) main shock
    ax.scatter(r0[0], r0[1],
           c = '#21ff60',
           alpha = 1,
           marker = "x")
    
    ax.set_ylabel('y position')
    ax.set_xlabel('x position')
    
    # formatting choices depending on whether viewing by aftershock generation/by time
    if color == "Generation":
        ax.set_facecolor('#1b2330')
        ax.set_title('Generated Events ({})'.format(total_events))
    elif color == "Time":
        ax.grid(True)
        ax.set_title('Generated Events ({})'.format(total_events))
    elif color == "Density":
        ax.set_title('Event Density by kNN, k = {}'.format(k))

    plt.savefig('newberry.png',dpi=400)

def frequency_by_interval(x, nbins, density = False):
    """
    For a given array of x values, determine the frequency of elements within nbins equally spaced bins partitioning x
    Returns (x,y) coords where x -> bin center, y -> frequency of x 
    """
    
    if density:
        n_events = len(x)
    else:
        n_events = 1
        
    edges = np.linspace(x.min(), x.max(), nbins + 1) # edges of bins
    centers = 0.5*(edges[1:]+edges[:-1]) # bin centers
    
    # get frequency (density) of events within each interval/bin
    frequencies = np.array([sum(1/n_events for i in x if (i >= edges[j] and i <= edges[j+1])) for j in range(nbins)])
    
    return centers, frequencies
    
    

def catalog_plots(catalog_pkl):
    catalogs = catalog_pkl[catalog_pkl.Magnitude != 0] # filter out for non-zero magnitude events
    catalogs = catalogs.sort_values(by = ['Time']) # sort in ascending order by time
    
    catalogs = catalogs.loc[:,['Magnitude','Time','Distance']]
    
    time = np.array(catalogs.Time, dtype = float)
    magnitude = np.array(catalogs.Magnitude)
    distance = np.array(catalogs.Distance)
    
    dt = np.array([np.abs(i-j) for i in time for j in time if i is not j]).min()
    dt = 0.95 * dt
    
    edges = np.concatenate((np.array([0]),time + dt))
    rates = [1/(edges[i]-edges[i-1]) for i in range(1,len(time)+1)]
    
    f, (ax1, ax2, ax4, ax6, ax8) = plt.subplots(5, figsize=(9,15))
    
    # plot seismicity rate
    plt.sca(ax1)
    ax1.plot(time, rates, color = 'orange')
    ax1.set_yscale('log')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Fequency of events per unit time')
    ax1.set_title('Seismicity Rate')
    plt.xlim([0, time.max()])
    plt.tight_layout()
    
    # plot event magnitude and event density with time
    plt.sca(ax2)
    markerline, stemlines, baseline = ax2.stem(time, magnitude, label = 'Events')
    ax2.set_yscale('log')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Event magnitude with time')
    plt.xlim([0, time.max()])
    
    ax3 = ax2.twinx()
    plt.sca(ax3)
    plt.grid(False)
    nbins = int(time.max()/3.5)
    bin_centers, frequencies = frequency_by_interval(time, nbins, density = True)
    sigma = 1
    bin_gauss = gaussian_filter1d(bin_centers, sigma)
    freq_gauss = gaussian_filter1d(frequencies, sigma)
    ax3.plot(bin_gauss, freq_gauss, color = 'black', label = 'Density')
    ax3.set_ylabel('Smoothed event density')
    plt.xlim([0, time.max()])
    
    # plot magnitude and event density with distance
    plt.sca(ax4)
    markerline, stemlines, baseline = ax4.stem(distance, magnitude)
    ax4.set_yscale('log')
    ax4.set_xlabel('Distance')
    ax4.set_ylabel('Magnitude')
    ax4.set_title('Event magnitude with distance')
    plt.xlim([0, distance.max()])
    plt.setp(stemlines, color = 'g')
    plt.setp(markerline, color = 'g')
    
    ax5 = ax4.twinx()
    plt.sca(ax5)
    plt.grid(False)
    nbins = int(distance.max()/3.5)
    bin_centers, frequencies = frequency_by_interval(distance, nbins, density = True)
    sigma = 0.8
    bin_gauss = gaussian_filter1d(bin_centers, sigma)
    freq_gauss = gaussian_filter1d(frequencies, sigma)
    ax5.plot(bin_gauss, freq_gauss, color = 'black')
    ax5.set_ylabel('Smoothed event density')
    plt.xlim([0, distance.max()])
    
    # plot k nearest neighbour density for time
    k = 9 # number of nearest neighbours
    plt.sca(ax6)
    markerline, stemlines, baseline = ax6.stem(time, magnitude, label = 'Events')
    ax6.set_yscale('log')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Magnitude')
    ax6.set_title('Event magnitude with time - kNN event density, k = {}'.format(k))
    plt.xlim([0, time.max()])
    
    ax7 = ax6.twinx()
    npoints = 200
    timegrid = np.linspace(time.min(), time.max(), npoints)
    time_list = list(time)
    kNN_density = np.array([k/kNN_measure(time_list, ti, k) for ti in timegrid])
    ax7.plot(timegrid, kNN_density, color = 'red')
    ax7.set_ylabel('Event density')
    ax7.set_yscale('log')
    
    # plot k nearest neighbour density for distance
    plt.sca(ax8)
    markerline, stemlines, baseline = ax8.stem(distance, magnitude, label = 'Events')
    ax8.set_yscale('log')
    ax8.set_xlabel('Distance')
    ax8.set_ylabel('Magnitude')
    ax8.set_title('Event magnitude with distance - kNN event density, k = {}'.format(k))
    plt.xlim([0, time.max()])
    plt.setp(stemlines, color = 'g')
    plt.setp(markerline, color = 'g')
    
    ax9 = ax8.twinx()
    plt.sca(ax9)
    distgrid = np.linspace(distance.min(), distance.max(), npoints)
    dist_list = list(distance)
    kNN_density = np.array([k/kNN_measure(dist_list, di, k) for di in distgrid])
    ax9.plot(distgrid, kNN_density, color = 'red')
    ax9.set_ylabel('Event density')
    ax9.set_yscale('log')
    
    
    plt.show()

def kNN_measure(x, x0, k, dim = 2):
    """
    Inputs:
    x -> list of np.array objects of same dimension or list of scalars
    x0 -> np.array or scalar
    k -> k nearest neighbours
    dim -> dimension of the vector space of elements in x
    
    Outputs:
    measure -> the distance/radius spanned by the k nearest neighbours of x0
    """
    # copy the x list so it doesn't get modified
    xcopy = x.copy()
    
#    neighbours = []
    neighbour_distances = []
    for j in range(k):
        distances = [(np.linalg.norm(xi-x0)) if xi is not x0 else np.inf for xi in xcopy]
        i = distances.index(min(distances)) # argmin{distances}
        neighbour_distances.append(min(distances))
        xcopy.pop(i)
    if dim == 2:
#        # remove any inf or nan
#        keep = [not(i == np.nan or i == np.inf) for i in neighbour_distances]
#        if False in keep:
#            neighbour_distances = compress(neighbour_distances, keep)
        measure = max(neighbour_distances)
        if measure == 0: # temporary
            measure = 0.1
    elif dim == 1:
        measure = np.abs(max(neighbour_distances) - min(neighbour_distances))
    return measure

def plot_ED(catalogs_raw, k = 4, plot = True):
    """
    Plot event density w.r.t distance from main shock. Also returns distance and density (x,y)
    Calculates densities by k-NN binning
    
    Inputs:
    catalogs_raw -> Pandas DataFrame event catalog
    """
    
    catalogs = catalogs_raw[catalogs_raw.Magnitude != 0] # extract events
    catalogs = catalogs.sort_values(by = ['Distance_from_origin']) # sort by distance
    
    x = catalogs.x
    y = catalogs.y
    distance = np.array(catalogs.Distance_from_origin, dtype = float) # get event distance from origin
    n = len(distance) # total number of events
    
    # get positions as list of numpy vectors
    positions = [np.array(([xi],[yi])) for xi,yi in zip(x,y)]
    density = np.array([k / (2 * n * kNN_measure(positions, event, k))  for event in positions], dtype = float) # get the kNN density for each event
    
    if plot:
        f, ax = plt.subplots(1, figsize=(7,6))
        ax.plot(distance, density, 'o')
        ax.set_yscale('log')#, nonposy = 'clip')
        ax.set_xlabel('distance from main shock')
        ax.set_ylabel('event density')
        ax.set_ylim(0,density.max())
        ax.set_xscale('log')#, nonposx = 'clip')
        plt.show()
    
    return distance, density
    
#    
def hav(lat1,lat2,long1,long2):
    '''
    Determine the haversine of the central angle between two points on a sphere given as latitudes and longitudes
    '''
    return 0.5*(1-np.cos(lat2-lat1)) + np.cos(lat1)*np.cos(lat2)*0.5*(1-np.cos(long2-long1))
    
def gcdist(R,lat1,lat2,long1,long2, deg = False):
    '''
    Returns the great circle distance between two points on a sphere, given their latitude and longitudes, and radius of sphere
    '''
    
    if deg:
        lat1, lat2, long1, long2 = lat1*np.pi/180, lat2*np.pi/180, long1*np.pi/180, long2*np.pi/180
    
    return 2*R*np.arcsin((hav(lat1,lat2,long1,long2))**0.5)

def rho(r, rho0, rc, gmma):
    '''
    Return the functional form for event density fall-off as described by Goebel, Brodsky
    '''
    return rho0 * 1/(1+(r/rc)**(2*gmma))**0.5

def rho2(r, rho0, rc, gmma):
    '''
    verifying my integral for rho
    '''
    a = np.array([float(mp.hyp2f1(0.5,0.5/gmma,1+0.5/gmma,-(ri/rc)**(2*gmma))) for ri in r])
    b = -r**(2*gmma)/(2+4*gmma)*np.array([float(mp.hyp2f1(1.5,(2*gmma+1)/(2*gmma), 2+0.5/gmma, -(ri/rc)**(2*gmma))) for ri in r])*(2*gmma/rc**(2*gmma))
    
    return rho0*(a + b)

def LLK_rho(theta,*const):
    '''
    Log likelihood function for radial event density.
    Inputs:
    theta -> 1d array-like of parameters in order rc, gmma
    *const -> (required) arguments specifying rho0, rmax, r vector
    Outupts:
    llk -> log likelihood, function of parameters
    '''
    rc, gmma = theta
    rmax, rmin, r, rho0, bin_edges, n_edges = const
    
#    r = r/r.max()
#    rmax = r.max()
##    sumr = np.sum(r**0.02)
##    N = len(r)
#    intgrl = rho0 * rmax * float(mp.hyp2f1(0.5,0.5/gmma,1+0.5/gmma,-(rmax/rc)**(2*gmma)))
#    sumlog = np.sum(np.log(rho(r, rho0, rc, gmma)))
#    llk = sumlog - intgrl
    
#    bin_edges = np.linspace(np.log(rmin), np.log(rmax), n) # fit from where data starts and ends, bins to be logarithmically spaced
#    bin_edges = np.exp(bin_edges) # back transform
#    bin_width = bin_edges[1] - bin_edges[0]
#    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    llk = 0
    for i in np.arange(n_edges-1):
        nobs = len(np.intersect1d(r[r>=bin_edges[i]], r[r<bin_edges[i+1]]))
        nobs = max(1,nobs)
        nexp = np.pi * (bin_edges[i+1]**2 - bin_edges[i]**2) * 1/(bin_edges[i+1]-bin_edges[i])*integrate.quad(rho, bin_edges[i], bin_edges[i+1], args = (rho0, rc, gmma))[0] #integrate.quad(rho, bin_edges[i+1], bin_edges[i], args = (rho0, rc, gmma))[0] * bin_width  #rho0 * bin_edges[i+1] * float(mp.hyp2f1(0.5,0.5/gmma,1+0.5/gmma,-(bin_edges[i+1]/rc)**(2*gmma))) - rho0 * bin_edges[i] * float(mp.hyp2f1(0.5,0.5/gmma,1+0.5/gmma,-(bin_edges[i]/rc)**(2*gmma))) # 
        nexp = max(np.finfo(float).eps, nexp)
#        if nexp <= 0:
#            print('hold up, nexp = {}'.format(nexp))
#            print('theta = {}, {}'.format(theta[0], theta[1]))
#            print('integral_analytic = {}'.format(rho0 * bin_edges[i+1] * float(mp.hyp2f1(0.5,0.5/gmma,1+0.5/gmma,-(bin_edges[i+1]/rc)**(2*gmma))) - rho0 * bin_edges[i] * float(mp.hyp2f1(0.5,0.5/gmma,1+0.5/gmma,-(bin_edges[i]/rc)**(2*gmma)))))
        llk += nobs * log(nexp) - nexp - log(np.math.factorial(nobs))#(nobs*log(nobs) - nobs + 1)
    
    return -llk

def gnom_x(lat, long, lat0, long0, deg = True):
    
    if deg:
        lat0,long0,lat,long = lat0*np.pi/180, long0*np.pi/180, lat*np.pi/180, long*np.pi/180
    
    cosc = np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(long-long0)
    
    x = np.cos(lat)*np.sin(long-long0) / cosc
    
    return x

def gnom_y(lat, long, lat0, long0, deg = True):
    
    if deg:
        lat0,long0,lat,long = lat0*np.pi/180, long0*np.pi/180, lat*np.pi/180, long*np.pi/180
    
    cosc = np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(long-long0)
    
    y = ( np.cos(lat0)*np.sin(lat) - np.sin(lat0)*np.cos(lat)*np.cos(long-long0) ) / cosc
    
    return y