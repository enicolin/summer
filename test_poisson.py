# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:07:54 2018

@author: enic156
"""

# test poisson sampler

import eq_functions as eq
import random as rnd
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
from scipy.stats import chisquare

np.random.seed(1756)
rnd.seed(a=1756)

nsamples = int(1e5)


#for lmbd in range(100,105):
#    # alternate colour of plots for easier readability
#    if lmbd % 2 == 0:
#        colour = "darkblue"
#    else:
#        colour = "red"
#    
#    poiss_mine = eq.sample_poisson(lmbd,nsamples)
#    
#    poiss_builtin = np.random.poisson(lmbd,nsamples)
#    
#    fig, axes = plt.subplots(2, 1)
#    
#    xlim = max(poiss_builtin.max(),poiss_mine.max()) # x limit for equal scale on both subplots
#    
#    # chi-squared test for how poisson my sampler is
#
#    # get frequency of observations for each poisson number
#    n_poissnumbers = int(poiss_mine.max()) + 1 # number of poisson numbers
#    f_obs = np.zeros(n_poissnumbers) # array to store frequency of observations of each sampled number
#    obs_counter = Counter(poiss_mine) # count occurences of each sampled poisson number
#    for i in range(n_poissnumbers): 
#        f_obs[i] = obs_counter[i] # in the ith position, store the number of occurences of the ith poisson number
#    
#    # get frequency of expected observations 
#    f_exp = np.zeros(n_poissnumbers)
#    for i in range(n_poissnumbers):
#        f_exp[i] = nsamples * eq.poisson(i, lmbd)
#    
#    chisq_mine, p_mine = chisquare(f_obs,f_exp,ddof=2)
#    
#    axes[0].hist(poiss_mine,color=colour,edgecolor="black",bins=list(np.linspace(0,int(poiss_mine.max()),int(poiss_mine.max())+1)))
#    axes[0].set_title('own sampler, {} = {}, p-value: {}'.format("$\lambda$", lmbd, p_mine))
#    axes[0].set_xlim([0,xlim])
#    
#    # chi-squared test for how poisson the built in sampler is
#
#    # get frequency of observations for each poisson number
#    n_poissnumbers = int(poiss_builtin.max()) + 1 # number of poisson numbers
#    f_obs = np.zeros(n_poissnumbers) # array to store number of observations of each sampled number
#    obs_counter = Counter(poiss_builtin) # count occurences of each sampled poisson number
#    for i in range(n_poissnumbers): 
#        f_obs[i] = obs_counter[i] # in the ith position, store the number of occurences of the ith poisson number
#    
#    # not super efficient, repeating some stuff uneccessarily here
#    # get frequency of expected observations for each poisson number
#    f_exp = np.zeros(n_poissnumbers)
#    for i in range(n_poissnumbers):
#        f_exp[i] = nsamples * eq.poisson(i, lmbd)
#    
#    chisq_builtin, p_builtin = chisquare(f_obs,f_exp,ddof=2)
#    
#    axes[1].hist(poiss_builtin,color=colour,edgecolor="black",bins=list(np.linspace(0,int(poiss_builtin.max()),int(poiss_builtin.max())+1)))
#    axes[1].set_title('built-in sampler, {} = {}, p-value = {}'.format("$\lambda$", lmbd, p_builtin))
#    axes[1].set_xlim([0,xlim])
#    plt.tight_layout()
#    plt.show()

q = 1000
pvalues = np.empty(q)
lmbd = 10
for b in range(q):
    poiss_mine = np.random.poisson(lmbd,nsamples)
    # get frequency of observations for each poisson number
    n_poissnumbers = int(poiss_mine.max()) + 1 # number of poisson numbers
    f_obs = np.zeros(n_poissnumbers) # array to store frequency of observations of each sampled number
    obs_counter = Counter(poiss_mine) # count occurences of each sampled poisson number
    for i in range(n_poissnumbers): 
        f_obs[i] = obs_counter[i] # in the ith position, store the number of occurences of the ith poisson number
    
    # get frequency of expected observations 
    f_exp = np.zeros(n_poissnumbers)
    for i in range(n_poissnumbers):
        f_exp[i] = nsamples * eq.poisson(i, lmbd)
    
    chisq_mine, p_mine = chisquare(f_obs,f_exp,ddof=2)
    pvalues[b] = p_mine
plt.hist(pvalues,bins=30)
plt.show()
