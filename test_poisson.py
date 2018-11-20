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

lmbd = 10

rand1 = eq.sample_poisson(lmbd,2000)

rand2 = np.random.poisson(lmbd,2000)

fig, axes = plt.subplots(2, 1)

axes[0].hist(rand1,color="darkblue",edgecolor="black",bins=50)
axes[0].set_title('own sampler')
axes[0].set_xlim([0,16])
axes[1].hist(rand2,color="darkblue",edgecolor="black",bins=50)
axes[1].set_title('built-in sampler')
axes[1].set_xlim([0,16])
plt.tight_layout()
plt.show()
