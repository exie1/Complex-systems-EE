#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:51:42 2020

@author: shni2598
"""

import brian2.numpy_ as np
#from brian2.only import *
import matplotlib.pyplot as plt

from connection import coordination


def input_spkrate(maxrate = [800,800], sig=[6,6], position=[[-32, -32],[0, 0]], n_side=64, width=64):
    '''
    generate the firing rate for the input poisson spike, the poisson rate of each input
    poisson spike has gaussian shape profile across the network
    position: the coordinate of the two peaks of two input stimuli
    maxrate: max rate of poisson input
    sig: standard deviation of the gaussian profile of input stimuli
    '''
    hw = width/2
    step = width/n_side
    x = np.linspace(-hw + step/2, hw - step/2, n_side)
    y = np.linspace(hw - step/2, -hw + step/2, n_side)
    
    xx, yy = np.meshgrid(x,y)
    
    lattice = np.zeros([n_side*n_side, 2])
    lattice[:,0] = xx.flatten()
    lattice[:,1] = yy.flatten()
      

    n_stim = len(position)
    for i in range(n_stim):
        dist_sti = coordination.lattice_dist(lattice, width, position[i])
        if i == 0:
            rate_sti = np.zeros(len(dist_sti))
        rate_sti += maxrate[i]*np.exp((-0.5)*(dist_sti/sig[i])**2)

    return rate_sti 


def input_adaptation(maxdelta = [10,5], sig=[6,6], position=[[-32, -32],[0, 0]], n_side=64, width=64):
    # lattice = coordination.makelattice(n_side,width,[0,0])
    hw = width/2
    step = width/n_side
    x = np.linspace(-hw + step/2, hw - step/2, n_side)
    y = np.linspace(hw - step/2, -hw + step/2, n_side)
    
    xx, yy = np.meshgrid(x,y)
    
    lattice = np.zeros([n_side*n_side, 2])
    lattice[:,0] = xx.flatten()
    lattice[:,1] = yy.flatten()
      
    n_stim = len(position)
    for i in range(n_stim):
        dist_sti = coordination.lattice_dist(lattice, width, position[i])
        if i == 0:
            delta_sti = np.zeros(len(dist_sti))
        delta_sti += maxdelta[i]*np.exp((-0.5)*(dist_sti/sig[i])**2)
    return delta_sti 