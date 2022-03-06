# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:08:04 2019

@author: nishe
"""

import numpy as np
#import matplotlib.pyplot as plt
#%%
def lattice(width, n, centre):
    #n = 1000
    #width = 62
    
    w = width + 1
    
    y = np.random.uniform(low = -w/2, high = w/2, size = n) + centre[1]
    y[::-1].sort()
    x = np.zeros(y.shape)
    
    seg = np.linspace(w/2, -w/2, w+1) + centre[1]
    ind = 0
    
    for i in range(0, len(seg)-1):
        
        y_seg = y[(y<seg[i]) & (y>=seg[i+1])]
        x_seg = np.random.uniform(-w/2, w/2, len(y_seg)) + centre[0]
        sort_ind = np.argsort(x_seg)
        x_seg.sort()
        x[ind:ind+len(x_seg)] = x_seg
        y[(y<seg[i]) & (y>=seg[i+1])] = y_seg[sort_ind]
        ind += len(x_seg)
        
    lattice = np.zeros([n,2])
    lattice[:,0] = x
    lattice[:,1] = y
    
    return lattice

#plt.scatter(x,y)
#plt.ylim(-31.5,31.5)



