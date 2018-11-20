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
    
    axes[0].hist(rand1,color="darkblue",edgecolor="black",bins=int(3*rand1.max()))
    axes[0].set_title('own sampler, {} = {}'.format("$\lambda$",lmbd))
    axes[0].set_xlim([0,20])
    axes[1].hist(rand2,color="darkblue",edgecolor="black",bins=int(3*rand2.max()))
    axes[1].set_title('built-in sampler, {} = {}'.format("$\lambda$",lmbd))
    axes[1].set_xlim([0,20])
    plt.tight_layout()
    plt.show()
