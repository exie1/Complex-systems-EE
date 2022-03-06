#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:31:53 2021

@author: shni2598
"""

import numpy as np
#from brian2.only import *
import matplotlib.pyplot as plt

from connection import coordination
#%%
def get_adaptation(base_amp = 16, max_decrease = [13,13], sig=[7,7], position=[[0, 0],[-32, -32]], n_side=64, width=64):
    '''
    calculate value of adaptation(delta_gk) with gaussian profile for each neuron
    base_amp: base-line amplitude of adaptation
    position: the coordinate of the modulation locations, modulation will decrease adaptation at corresponding locations.
    max_decrease: max decrease from 'base_amp' due to modulation
    sig: standard deviation of the gaussian profile of adaptation modulation
    '''
    # #width = 62
    # hw = width/2
    # #n_side = 63
        
    # x = np.linspace(-hw,hw,n_side) #+ centre[0]
    # y = np.linspace(hw,-hw,n_side) #+ centre[1]
    hw = width/2
    step = width/n_side
    x = np.linspace(-hw + step/2, hw - step/2, n_side)
    y = np.linspace(hw - step/2, -hw + step/2, n_side)
    
    xx, yy = np.meshgrid(x,y)
    
    lattice = np.zeros([n_side*n_side, 2])
    lattice[:,0] = xx.flatten()
    lattice[:,1] = yy.flatten()
      
    #sti1 = [-31.5, -31.5]
    #sti2 = [0, 0]
    n_stim = len(position)
    for i in range(n_stim):
        dist = coordination.lattice_dist(lattice, width, position[i])
        if i == 0:
            adapt_dec = np.zeros(len(dist))
        adapt_dec += max_decrease[i]*np.exp((-0.5)*(dist/sig[i])**2)
    
    adapt = base_amp - adapt_dec
#    dist_sti1 = coordination.lattice_dist(lattice, width, position[0])
#    dist_sti2 = coordination.lattice_dist(lattice, width, position[1])
#    
#    sig1 = sig2 = sig#(width+1)*(0.6/(2*np.pi))
#    maxr1 = maxr2 = maxrate
#    rsti1 = maxr1*np.exp((-0.5)*(dist_sti1/sig1)**2)
#    rsti2 = maxr2*np.exp((-0.5)*(dist_sti2/sig2)**2)
    
    return adapt #rsti1+rsti2
#%%
#a=input_spkrate()
#rate = input_spkrate(maxrate=[600,800], sig=[2,6])
#rate = rate.reshape(63,63)
##%%
#plt.figure()
#plt.imshow(rate)
#plt.colorbar()

