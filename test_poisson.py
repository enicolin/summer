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

rnd.seed(a=1756)

#lmbd = 1


for lmbd in range(1,15):
    rand1 = eq.sample_poisson(lmbd,2000)
    
    rand2 = np.random.poisson(lmbd,2000)
    
    fig, axes = plt.subplots(2, 1)
    
    axes[0].hist(rand1,color="darkblue",edgecolor="black",bins=list(np.linspace(0,int(rand1.max()),int(rand1.max())+1)))
    axes[0].set_title('own sampler, {} = {}'.format("$\lambda$",lmbd))
    axes[0].set_xlim([0,rand1.max()])
    axes[1].hist(rand2,color="darkblue",edgecolor="black",bins=list(np.linspace(0,int(rand2.max()),int(rand2.max())+1)))
    axes[1].set_title('built-in sampler, {} = {}'.format("$\lambda$",lmbd))
    axes[1].set_xlim([0,rand2.max()])
    plt.tight_layout()
    plt.show()
