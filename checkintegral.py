# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 08:26:39 2019

@author: enic156
"""

import eq_functions as eq
import numpy as np
import matplotlib.pyplot as plt 

a = np.random.uniform(1,5)
b = np.random.uniform(1,5)
c = np.random.uniform(1,5)

r = np.linspace(0,10,75)
y1 = eq.rho(r, a, b, c)
y2 = eq.rho2(r, a, b, c)

f, ax = plt.subplots(1, figsize = (7,7))

ax.plot(r, y1, color = 'r')
ax.plot(r, y2, 'o', color = 'k')
ax.set_xscale('log')
ax.set_yscale('log')

plt.show()