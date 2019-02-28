import numpy as np
import random as rnd
from math import log
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial
#from scipy import integrate
from scipy.ndimage.filters import gaussian_filter1d
import mpmath as mp
from scipy.special import erf
from scipy.special import exp1 as W
#from mpl_toolkits.axes_grid1 import make_axes_locatable
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
        return("{}(mgn = {}, time = {}, x = {}, y = {}, dst = {}, dst_frm_orgn = {}, gen = {})".format(self.__class__.__name__,
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
    
    dm = 1. # difference between main shock and greatest aftershock according to Bath
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
        

def generate_catalog(t0, r0, catalog_list, gen,
                     Tf,M0,A,alpha,b,c,cprime,p,pprime,Mc,smin, recursion = True):
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
                                 catalog_list, gen+1,
                                 Tf,parent_shocks.iat[i,2],A,alpha,b,c,cprime,p,pprime,Mc,smin,recursion=recursion)
    else:
        catalog_list.append(catalog)
        return

def plot_catalog(catalogs_raw, M0, r0, color = 'Time', savepath = None, saveplot = False, k = 20):
    """
     Plots generated synthetic catalog from generate_catalog
     Inputs:
     catalogs_raw -> concatenated output list of pandas DataFrames from generate_catalog
     M0 -> main shock magnitude
     color -> color scheme of events
               'Time' - default, colours events by time of occurrence
               'Generation' - colours by aftershock generation
    """
    f, ax = plt.subplots(1, figsize=(8,8))#, constrained_layout = True)
  
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
#        cmax = times.max()
        plt.ylim(np.concatenate((y,[r0[1]])).min(),np.concatenate((y,[r0[1]])).max())
        plt.xlim(np.concatenate((x,[r0[0]])).min(),np.concatenate((x,[r0[0]])).max())
        plot = ax.scatter(x, y,
                   c = c,
                   s = 0.05*10**magnitudes, # large events displayed much bigger than smaller ones
                   cmap = cmap,
                   alpha = 0.75)
#        plt.clim(0, cmax)
        
#        cax = f.add_axes([1, 0, 0.1, 1])
#        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, fraction=0.046, pad=0.04,label=color)
#        cbar.set_label(color)
    elif color == 'Generation':
        colors = ['#ffa220', '#ddff20', '#20fffb', '#866be0', '#d83a77'] # haven't seen more than 4 generations, so up to 5 colours should be fine
        # filter events by generation - in order to provide legend for aftershock generation
        catalogs_bygen = [] # store them in a list
        for i in range(catalogs['Generation'].max()+1):
            catalogs_bygen.append(catalogs[catalogs.Generation == i])
        
        for catalog, event_color in zip(catalogs_bygen, colors):
            # get coordinates
            x = catalogs['x']
            y = catalogs['y']
            
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
        npoints = 120
#        positions = [np.array(([xi],[yi])) for xi, yi in zip(x,y)]
        positions = list(zip(x.ravel(), y.ravel()))
        xgrid = np.linspace(x.min(), x.max(), npoints)
        ygrid = np.linspace(y.min(), y.max(), npoints)
#        points = [np.array(([xi,yi])) for yi in ygrid for xi in xgrid] # points at which to sample kNN
        xm, ym = np.meshgrid(xgrid, ygrid)
        points = list(zip(xm.ravel(), ym.ravel()))
        density = np.zeros(np.shape(xm))
        p = 0
        point_tree = spatial.KDTree(positions)
#        r = 100
        for i in range(len(xgrid)):
            for j in range(len(ygrid)):
                density[i][j] = k/(np.pi*(point_tree.query(points[p], k = k)[0][-1]))**2#k/(2 * total_events * kNN_measure(positions, points[p], k, dim = 2))
                p += 1
        plot = plt.contourf(xm, ym, density, 200, cmap = 'plasma')
        max_seism = np.unravel_index(density.argmax(), density.shape)
        plt.plot(xm[max_seism[0]][max_seism[1]],ym[max_seism[0]][max_seism[1]],marker='x',color='r')
        
    # plot the (initial/parent of all parents) main shock
    ax.scatter(r0[0], r0[1],
           c = '#21ff60',
           alpha = 1,
           marker = "v")
    
    plt.ylim(np.concatenate((y,[r0[1]])).min(),np.concatenate((y,[r0[1]])).max())
    plt.xlim(np.concatenate((x,[r0[0]])).min(),np.concatenate((x,[r0[0]])).max())
    
    ax.set_ylabel('y position')
    ax.set_xlabel('x position')
    
    # formatting choices depending on whether viewing by aftershock generation/by time
    if color == "Generation":
        ax.set_facecolor('#1b2330')
        ax.set_title('Generated Events ({})'.format(total_events))
    elif color == "Time":
#        ax.grid(True)
        ax.set_title('Generated Events ({})'.format(total_events))
    elif color == "Density":
        ax.set_title('Event Density by kNN, k = {}, {} events'.format(k, total_events))
    
    plt.show()
    if saveplot:
        plt.savefig(savepath,dpi=400)
        plt.close('all')

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
    ax1.set_ylabel('Frequency of events per unit time')
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
#    ax5.set_yscale('log')
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

def kNN_measure(x, x0, k, goebel_dens = False, dim = 2):
    """
    Note: this is essentially a brute force method. Instead use KD-trees for
    querying nearest neighbours.
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
        rmin = min(neighbour_distances)
        rmax = max(neighbour_distances)
        measure = rmax - rmin
    elif dim == 1:
        measure = np.abs(max(neighbour_distances) - min(neighbour_distances))
    if goebel_dens:
        return np.pi*(rmax- rmin)**2
    else:
        return measure

def plot_ED(catalogs_raw, k = 20, plot = True):
    """
    Plot event density w.r.t distance from main shock and return distance and density (x,y)
    Calculates densities by k-NN binning
    
    Inputs:
    catalogs_raw -> Pandas DataFrame event catalog
    k -> number of nearest neighbours to query - affects smoothness of estimates
    """
    
    catalogs = catalogs_raw[catalogs_raw.Magnitude != 0] # extract events
    catalogs = catalogs.sort_values(by = ['Distance_from_origin']) # sort by distance
    
#    x = catalogs.x
#    y = catalogs.y
    r = np.array(catalogs.Distance_from_origin, dtype = float) # get event distance from origin
    
#
#    positions = list(zip(x.ravel(), y.ravel()))
#    point_tree = spatial.KDTree(positions)
#    density1 = np.array([k / (np.pi*(point_tree.query(event, k = k)[0][-1]))**2 for event in positions], dtype = float)
    
    n = len(r) # total number of events
#    positions = list(zip(r,np.zeros(n)))
#    dist_tree = spatial.KDTree(positions)
#    ind = [dist_tree.query(event, k = k)[1] for event in positions] # get indices of kNN for ith event
##    kNN_dist = np.array([((dist_tree.query(event, k = k)[0][-1] - dist_tree.query(event, k = k)[0][0])) for event in positions])
#    kNN_dist = np.array([r[max(indi)] - r[min(indi)] for indi in ind])
#    density0 = k/(np.pi*kNN_dist**2) # density prior to rolling average
##    density = np.zeros(n)
#
    density0 = np.zeros(n)
    positions = np.array([[i] for i in r])
    dist_tree = spatial.KDTree(positions)
    for i, ri in enumerate(r):
        dist, ind = dist_tree.query(np.array([[ri]]), k = k)
        density0[i] = k/(np.pi*(float(r[ind.max()]**2 - r[ind.min()]**2)))


#    density0 = np.zeros(n)
#    positions = np.array([[i] for i in r])
#    dist_tree = spatial.KDTree(positions)
#    for i in range(int(n-k/2)):
#        ind = [dist_tree.query(np.array([[rj]]), k = k)[1] for rj in r[int(i):int(i+k)]]
#        density0[i] = k/(np.pi * np.mean([float(r[ii.max()] - r[ii.min()])**2 for ii in ind]))                #k/(np.pi * (float(r[ind.max()] - r[ind.min()]))**2)

#    for j in range(int(0),int(n-k)):
#        denswindow = (density0[int(j):int(j+k)])
#        density[j] = np.mean(denswindow) #np.mean(denswindow/distance[int(j-k/2):int(j+k/2)]**2)

    if plot:
        f, ax = plt.subplots(1, figsize=(7,6))
        ax.plot(r, density0, 'o')
        ax.set_yscale('log')#, nonposy = 'clip')
        ax.set_xlabel('distance from main shock')
        ax.set_ylabel('event density')
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.show()

    return r, density0
     
def hav(lat1,lat2,long1,long2):
    '''
    Determine the haversine of the central angle between two points on a sphere given as latitudes and longitudes
    '''
    return 0.5*(1-np.cos(lat2-lat1)) + np.cos(lat1)*np.cos(lat2)*0.5*(1-np.cos(long2-long1))
    
def gcdist(R,lat1,lat2,long1,long2, deg = False):
    '''
    Returns the great circle distance between two points on a sphere, given their latitude and longitudes, and radius of sphere.
    Assumes angles to be given in degrees by default, and in radians otherwise.
    Inputs:
        lat1, lat2 -> latitude of point 1 and 2
        long1, long2 -> longitude of point 1 and 2
    '''
    
    if deg:
        lat1, lat2, long1, long2 = lat1*np.pi/180, lat2*np.pi/180, long1*np.pi/180, long2*np.pi/180
    
    return 2*R*np.arcsin((hav(lat1,lat2,long1,long2))**0.5)

def rho(r, rho0, rc, gmma):
    '''
    Return the functional form for event density fall-off as described by Goebel, Brodsky
    '''
#    dens = rho0/(1+(r/rc)**(2*gmma))**0.5
    dens = np.exp(np.log(rho0) - 0.5*np.log(1+(r/rc)**(2*gmma)))

    return dens

def rho2(r, rho0, rc, gmma):
    '''
    verifying my integral for rho
    '''
    a = np.array([float(mp.hyp2f1(0.5,0.5/gmma,1+0.5/gmma,-(ri/rc)**(2*gmma))) for ri in r])
    b = -r**(2*gmma)/(2+4*gmma)*np.array([float(mp.hyp2f1(1.5,(2*gmma+1)/(2*gmma), 2+0.5/gmma, -(ri/rc)**(2*gmma))) for ri in r])*(2*gmma/rc**(2*gmma))
    
    return rho0*(a + b)
    
def LLK_rho(theta,rmax, rmin, r, bin_edges, n_edges, rho0):
    '''
    Log likelihood function for radial event density. NOT RELIABLE
    Inputs:
    theta -> 1d array-like of parameters in order rc, gmma
    *const -> (required) arguments specifying rho0, rmax, r vector
    Outupts:
    llk -> log likelihood, function of parameters
    '''
    rc, gmma = theta
#    rmax, rmin, r, bin_edges, n_edges, rho0 = const
    
    if (rc > 900) or (rc < 0) or (gmma < 1) or (gmma > 6):
        return -np.inf

#    r0 = bin_edges[1]-bin_edges[0]
    llk = 0
    for i in range(1, n_edges-1):
        nobs = len(np.intersect1d(r[r>=bin_edges[i-1]], r[r<=bin_edges[i+1]]))
        if nobs <= 0:
            nobs = 1e-8
#        nobs = max(eps,nobs)
#        factor = 2 * np.pi # np.pi * (bin_edges[i+1] - bin_edges[i])**2
#        integral = abs(integrate.quad(rho, bin_edges[i-1], bin_edges[i+1], args = (rho0, rc, gmma))[0])
        r0 = bin_edges[i]
#        A = rho0/(1+(r0/rc)**(2*gmma))**0.5 # coefficients for linear approximation to density function
        B = (rho0*gmma*r0**(2**gmma-1))/(rc**(2*gmma)*(1+(r0/rc)**(2*gmma))**1.5)
        a = bin_edges[i-1]
        b = bin_edges[i+1]
        A = np.exp(log(rho0) - 0.5*log(rc**(2*gmma)+r0**(2*gmma)) + gmma*log(rc))
#        B = np.exp(log(rho0*gmma*r0**(2*gmma-1)) -log(rc**(2*gmma)*(1+(r0/rc)**(2*gmma))**1.5))
        integral = abs((A+B*r0)*(b-a) + 0.5*B*(a**2-b**2))
#        if integral <= 0:
#            return np.inf

        nexp = integral# * factor
        if nexp <= 0:
            nexp = 1e-8
            
        llk += (nobs * log((nexp)) - (nexp) - (nobs*log(nobs) - nobs + 1))
    
    return llk

def gnom_x(lat, long, lat0, long0, deg = True):
    '''
    Return x coordinate of a Gnomonic projection given a tangent plane's latitude/longitude coordinate where it contacts a unit sphere.
    Assumes angles to be in degrees by default, and radians otherwise.
    Inputs:
        lat, long -> latitude and longitude of point of contact of tangent plane to unit sphere (injection point coordinate in context)
        lat0, long0 -> latitude and longitude of point on unit sphere to project onto the tangent plane
    '''
    if deg:
        lat0,long0,lat,long = lat0*np.pi/180, long0*np.pi/180, lat*np.pi/180, long*np.pi/180
    
    cosc = np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(long-long0)
    
    x = np.cos(lat)*np.sin(long-long0) / cosc
    
    return x

def gnom_y(lat, long, lat0, long0, deg = True):
    '''
    Return y coordinate of a Gnomonic projection given a tangent plane's latitude/longitude coordinate where it contacts a unit sphere.
    Assumes angles to be in degrees by default, and radians otherwise.
    Inputs:
        lat, long -> latitude and longitude of point of contact of tangent plane to unit sphere (injection point coordinate in context)
        lat0, long0 -> latitude and longitude of point on unit sphere to project onto the tangent plane
    '''    
    if deg:
        lat0,long0,lat,long = lat0*np.pi/180, long0*np.pi/180, lat*np.pi/180, long*np.pi/180
    
    cosc = np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(long-long0)
    
    y = ( np.cos(lat0)*np.sin(lat) - np.sin(lat0)*np.cos(lat)*np.cos(long-long0) ) / cosc
    
    return y

def robj(prms, r, dens, bin_edges, q, MCMC):
    '''
    Objective function to be minimised for Goebel model fit. Weighted least squares function.
    '''
    rc, gmma, rho0 = prms
    
#    var = []
#    n_app = 0
#    for a, b, i in zip(bin_edges[:-1],bin_edges[1:], range(len(bin_edges)-1)):
#        m = q#len(np.intersect1d(r[r>=a], r[r<b])) # number of events in current bin interval
#        dens_i = dens[i*m:i*m+m]
#        var_i = np.std(np.log10(dens_i))**2 if np.std(np.log10(dens_i))**2 > 0 else 1e-8
#        for k in range(m):
#            var.append(var_i)
#            n_app += 1
#    var.append(var_i)
#    for k in range(len(dens)-n_app-1): # append remaining sigma at the end
#        var.append(var_i)
#    var = np.array(var)

    lb = [1, 1, 1e-4]
    ub = [1000, 6, 1]
    
    # bounds to specify for emcee
    if (rc < lb[0] or rc > ub[0]) or (gmma < lb[1] or gmma > ub[1]) or (rho0 < lb[2] or rho0 > ub[2]):
        return -np.inf
    
    exp = rho(r, rho0, rc, gmma)
    obj = -np.sum(((dens-exp)/dens)**2)
    
    if MCMC:
        return obj
    else:
        return -obj # psywarm minimises functions
    
def p1D(r,*args):
    '''
    Solution for maximum pressure for distant radius r in 1D.
    '''
    alpha, T, k, nu, q = args
    R = (4*alpha*T)**0.5
    arg = r/R
    p_factor = (2*np.pi*k)/(q*nu*(alpha*T)**0.5)
    p = p_factor*4*(np.pi)**0.5*((1+2*arg**2)**0.5*np.exp(-arg**2/(1+2*arg**2)) - 2**0.5*arg*np.exp(-0.5) - np.pi**0.5*arg*(1-erf(arg/(1+2*arg**2)**0.5)-(1-erf(1/2**0.5))))
    return p

def p2D(r, *args):
    '''
    Solution for maximum pressure for distant radius r in 2D.
    '''
    
    alpha, T, k, nu, q = args
    R = (4*alpha*T)**0.5
    arg_r = r/R
    p_factor = (4*np.pi*k)/(nu*q)
#    p_factor = (nu*q*R)/(4*np.pi*k) # ([m^2/s * kg/m^2/s]/[m^2])*[m]
    p = (W(1/(1+1/arg_r**2)) - W(1)) * p_factor
    return p

def p3D(r, *args):
    '''
    Solution for maximum pressure for distant radius r in 3D.
    '''
    alpha, T, k, nu, q = args
    R = (4*alpha*T)**0.5
    arg = r/R
    p_factor = (8*np.pi*k*(alpha*T)**0.5)/(nu*q)
    p = (erf((3/2)**0.5) - erf(arg/(1+2/3*arg**2)**0.5))/arg * p_factor
    return p

def p2D_transient(r, t_now, C, pc, *args):
    '''
    2D solution for pressure during and after a period of (constant) injection
    '''
    
    alpha, T, k, nu, q, rc, pnoise = args
#    mask = r >= rc # mask for the fit region
#    r_fit = r[mask]
#    r_plateau = r[~mask]
#    assert(len(r_fit) + len(r_plateau) == len(r))
    
    R = (4*alpha*T)**0.5 # characteristic length
    arg_t = t_now/T
    
    # for time after injection
    if arg_t >= 1:
        P = C*(4*np.pi*k)/(nu*q)
        rbf = (arg_t-1)**0.5
        mask = r >= min(rbf*R,rc) # mask for the fit region
        r_fit = r[mask]
        r_plateau = r[~mask]
        arg_r = r_fit/R # fit to region of 'large r'
        assert(len(r_fit) + len(r_plateau) == len(r))
        
        
        r_lwr = arg_r[arg_r < rbf] # pressure here is time invariant
        r_upr = arg_r[arg_r >= rbf] # transient pressure
        
        p_upr = ((W(r_upr**2/arg_t)-W(r_upr**2/(arg_t-1)))-pc) * P + pnoise# transient pressure - includes term for background seismicity and threshold pressure
        p_upr[np.where(p_upr<0)] = 0
        p_lwr = (W(1/(1+1/r_lwr**2)) - W(1) -pc) * P + pnoise  # pmax - includes term for background seismicity and threshold pressure
        p_lwr[np.where(p_lwr<0)] = 0
        p = np.concatenate((p_lwr,p_upr))
        
        p_plateau = np.array([p[0] for i in r_plateau])
        p = np.concatenate((p_plateau, p))
        assert(len(p) == len(r))
    else:
        P = C*(4*np.pi*k)/(nu*q)
#        P = 1/P
        mask = r >= rc # mask for the fit region
        r_fit = r[mask]
        r_plateau = r[~mask]
        arg_r = (r_fit)/R # fit to region of 'large r'
        
        p_fit = (W(arg_r**2/arg_t)-pc) * P + pnoise # transient pressure - includes term for background seismicity and threshold pressure
        p_plat = np.array([p_fit[0] for i in r_plateau])
        p = np.concatenate((p_plat, p_fit))
        p[np.where(p<0)] = 0
    return p
    
def robj_diff(theta, *const):
    '''
    Weighted least squares objective used for calibrating the 2D diffusion model in time.
    '''
    alpha, k, nu, q, rc, pc = theta[:6]
    C = theta[6:-1]
    pnoise = theta[-1]
    r, dens, MCMC, lb, ub, T, t_now = const
    
    # or (rc < lb[6] or rc > lb[6]) or (rbf_D < rc): 
#    if (alpha < lb[0] or alpha > ub[0]) or (T < lb[1] or T > ub[1]) or \
#    (k < lb[2] or k > ub[2]) or (nu < lb[3] or nu > ub[3]) or (q < lb[4]\
#    or q > ub[4]) or (t_now < lb[5] or t_now > ub[5]):
#        return -np.inf
    
    obj = 0
    for ri, densi, ti, Ci in zip(r, dens, t_now, C):
        exp = p2D_transient(ri, ti, Ci, pc, alpha, T, k, nu, q, rc, pnoise)
        obj += -np.sum(((densi-exp)/densi)**2)#*T/min(ti,T)
    
    if not MCMC:
        return -obj
    else:
        return obj