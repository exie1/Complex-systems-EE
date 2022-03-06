#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:41:54 2021

@author: shni2598
"""
import numpy as np
#from brian2.only import *
#import matplotlib.pyplot as plt

from connection import coordination
#%%
def get_adaptation(base_amp = 16, max_decrease = [13,13], rang = [6, 6], sharpness =[1,1], position=[[0, 0],[-32, -32]], n_side=64, width=64):
    
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
        #adapt_dec += max_decrease[i]*np.exp((-0.5)*(dist/sharpness[i])**2)
    
        adapt_dec_0 = (1/(1+np.exp(-(dist+rang[i])/sharpness[i])))*((1 - 1/(1+np.exp(-(dist-rang[i])/sharpness[i]))))
        adapt_dec += adapt_dec_0*max_decrease[i]/((1/(1+np.exp(-(rang[i])/sharpness[i])))*((1 - 1/(1+np.exp(-(-rang[i])/sharpness[i])))))

    
    
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
# import adapt_gaussian

# #%%

# adapt_logi = get_adaptation(base_amp = 16, max_decrease = [15,15], rang = [8, 8], sharpness =[1.7,1.7], position=[[0, 0],[-32, -32]], n_side=64, width=64)

# #%%
# plt.figure()
# plt.imshow(adapt_logi.reshape(64,64))

# #%%
# adapt_gaus = adapt_gaussian.get_adaptation(base_amp = 16, max_decrease = [15,15], sig=[7,7], position=[[0, 0],[-32, -32]], n_side=64, width=64)
# #%%
# fig, [ax1,ax2] = plt.subplots(1,2)
# im1 = ax1.imshow(adapt_logi.reshape(64,64))
# #plt.colorbar(im1)
# im2 = ax2.imshow(adapt_gaus.reshape(64,64))
# #plt.colorbar(im2)
# #%%
# plt.figure()
# plt.plot(adapt_logi.reshape(64,64)[31])
# plt.plot(adapt_gaus.reshape(64,64)[31])




