#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 21:12:37 2020

@author: shni2598
"""
import numpy as np

#%%
class get_stim_scale:
    
    def __init__(self):
        self.seed = 10
        self.stim_dura = 200 # ms
        self.separate_dura = np.array([300,500]) # ms
        self.dt_stim = 10 # ms
        self.stim_amp_scale = None
        
    def get_scale(self):
        stim_num = self.stim_amp_scale.shape[0]
        stim_dura = self.stim_dura//self.dt_stim;
        separate_dura = self.separate_dura//self.dt_stim
        np.random.seed(self.seed)
        sepa = np.random.rand(stim_num)*(separate_dura[1]-separate_dura[0]) + separate_dura[0]
        sepa = sepa.astype(int)
        self.scale_stim = np.zeros([int(round(stim_num*stim_dura+sepa.sum()))])#, n_neuron])
        self.stim_on = np.zeros([stim_num, 2], int) 
        for i in range(stim_num):
            self.scale_stim[i*stim_dura + sepa[:i].sum(): i*stim_dura + sepa[:i].sum()+stim_dura] = self.stim_amp_scale[i] #* stim_rate
            self.stim_on[i] = np.array([i*stim_dura + sepa[:i].sum(), i*stim_dura + sepa[:i].sum()+stim_dura]) * self.dt_stim
                
        pass