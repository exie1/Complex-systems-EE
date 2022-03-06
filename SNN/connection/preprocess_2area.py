#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:42:09 2020

@author: shni2598
"""

import numpy as np
#import brian2.numpy_ as np
#import matplotlib.pyplot as plt
import connection as cn
import warnings
from scipy import sparse
#from brian2.only import *
#import time
#import mydata
#import firing_rate_analysis
#import os
#import datetime
#%%
class get_ijwd_2:
    '''
    get connectiviy (pre/post-synaptic neurons, weight, and delay) of inter-area connections
    '''
    def __init__(self, *param_dict, **kwargs):
    
        self.Ne1 = 4096 # number of excitatory neurons in bottom/sensory area
        self.Ni1 = 1024 # number of inhibitory neurons in bottom/sensory area
        self.Ne2 = 4096 # number of excitatory neurons in top/association area
        self.Ni2 = 1024 # number of inhibitory neurons in top/association area
        self.width1 = 64 # width of the size of bottom/sensory area
        self.width2 = 64 # width of the size of top/association area
                
        self.p_inter_area_1 = 0.5;# 0~1; proportion of excitatory neurons in bottom area that can form bottom-up connections
        self.section_width_1 = 2 # width of sub-network for choosing pre-synaptic bottom-up projection neurons in bottom area. see method 'choose_interarea_projection_neurons' below. 
        self.p_inter_area_2 = 0.5; # 0~1; proportion of excitatory neurons in top area that can form top-down connections
        self.section_width_2 = 2 # width of sub-network for choosing pre-synaptic top-down projection neurons in top area. see method 'choose_interarea_projection_neurons' below. 
        
        '''peak probability and decay constant of exponential connection probability'''
        self.peak_p_e1_e2 = 0.8; self.tau_p_d_e1_e2 = 4.2 # bottom(1) exi-neurons to top(2) exi-neurons
        self.peak_p_e1_i2 = 0.8; self.tau_p_d_e1_i2 = 4.2 # bottom exi-neurons to top inh-neurons
        self.peak_p_e2_e1 = 0.35; self.tau_p_d_e2_e1 = 7 # top exi-neurons to bottom exi-neurons
        self.peak_p_e2_i1 = 0.35; self.tau_p_d_e2_i1 = 7 # top exi-neurons to bottom inh-neurons
        
        ''' mean synaptic weigth; in nsiemens'''
        self.w_e1_e2_mean = 5; self.w_e1_i2_mean = 5 # nsiemens
        self.w_e2_e1_mean = 5; self.w_e2_i1_mean = 5 # nsiemens

        self.delay = [8,10] # lower and upper bound of synaptic delay, unifomly distributed; millisecond

        self.periodic_boundary = True; 
        self.interarea_dist = 0; # distance between areas

    def generate_ijwd(self):
        
        '''coordination of neurons'''
        self.e_lattice1 = cn.coordination.makelattice(int(round(self.Ne1**0.5)), self.width1, [0, 0])
        self.e_lattice2 = cn.coordination.makelattice(int(round(self.Ne2**0.5)), self.width2, [0, 0])
        self.i_lattice1 = cn.coordination.makelattice(int(round(self.Ni1**0.5)), self.width1, [0, 0])
        self.i_lattice2 = cn.coordination.makelattice(int(round(self.Ni2**0.5)), self.width2, [0, 0])                
        
        '''index of pre-synaptic inter-area projection neurons'''
        self.inter_e_neuron_1 = self.choose_interarea_projection_neurons(self.p_inter_area_1, int(self.Ne1**0.5), int(self.Ne1**0.5), self.section_width_1)#np.random.choice(self.Ne1, int(self.p_inter_area_1*self.Ne1), replace = False)
        self.inter_e_neuron_1 = np.sort(self.inter_e_neuron_1)

        self.inter_e_neuron_2 = self.choose_interarea_projection_neurons(self.p_inter_area_2, int(self.Ne2**0.5), int(self.Ne2**0.5), self.section_width_2)#np.random.choice(self.Ne2, int(self.p_inter_area_2*self.Ne2), replace = False)
        self.inter_e_neuron_2 = np.sort(self.inter_e_neuron_2)

        ''' i: index of pre-synaptic neurons
            j: index of post-synaptic neurons
            dist: distance between neurons
            w: synaptic weights
            d: synaptic delay'''
        self.i_e1_e2, self.j_e1_e2, dist_e1_e2 = cn.connect_2lattice.expo_decay(self.e_lattice1, self.e_lattice2, self.inter_e_neuron_1, \
                                                        self.width2, self.periodic_boundary, self.interarea_dist, \
                                                        self.peak_p_e1_e2, self.tau_p_d_e1_e2, src_equal_trg = False, self_cnt = False)
        
        self.i_e1_i2, self.j_e1_i2, dist_e1_i2 = cn.connect_2lattice.expo_decay(self.e_lattice1, self.i_lattice2, self.inter_e_neuron_1, \
                                                        self.width2, self.periodic_boundary, self.interarea_dist, \
                                                        self.peak_p_e1_i2, self.tau_p_d_e1_i2, src_equal_trg = False, self_cnt = False)

        self.i_e2_e1, self.j_e2_e1, dist_e2_e1 = cn.connect_2lattice.expo_decay(self.e_lattice2, self.e_lattice1, self.inter_e_neuron_2, \
                                                        self.width1, self.periodic_boundary, self.interarea_dist, \
                                                        self.peak_p_e2_e1, self.tau_p_d_e2_e1, src_equal_trg = False, self_cnt = False)
        
        self.i_e2_i1, self.j_e2_i1, dist_e2_i1 = cn.connect_2lattice.expo_decay(self.e_lattice2, self.i_lattice1, self.inter_e_neuron_2, \
                                                        self.width1, self.periodic_boundary, self.interarea_dist, \
                                                        self.peak_p_e2_i1, self.tau_p_d_e2_i1, src_equal_trg = False, self_cnt = False)
        

        self.w_e1_e2 = np.abs(np.random.normal(self.w_e1_e2_mean, self.w_e1_e2_mean*0.1, len(self.i_e1_e2)))#*nS*scale_e_21[j]
        self.w_e1_i2 = np.abs(np.random.normal(self.w_e1_i2_mean, self.w_e1_i2_mean*0.1, len(self.i_e1_i2)))#*nS*scale_e_21[j]
        self.w_e2_e1 = np.abs(np.random.normal(self.w_e2_e1_mean, self.w_e2_e1_mean*0.1, len(self.i_e2_e1)))#*nS*scale_e_21[j]
        self.w_e2_i1 = np.abs(np.random.normal(self.w_e2_i1_mean, self.w_e2_i1_mean*0.1, len(self.i_e2_i1)))#*nS*scale_e_21[j]

        self.d_e1_e2 = np.random.uniform(self.delay[0], self.delay[1], len(self.i_e1_e2))
        self.d_e1_i2 = np.random.uniform(self.delay[0], self.delay[1], len(self.i_e1_i2))
        self.d_e2_e1 = np.random.uniform(self.delay[0], self.delay[1], len(self.i_e2_e1))
        self.d_e2_i1 = np.random.uniform(self.delay[0], self.delay[1], len(self.i_e2_i1))
    
    def choose_interarea_projection_neurons(self, p_inter_area, N_x_all, N_y_all, section_width):
        '''
        Divide each area to sub-networks with size of 'section_width'*'section_width', and then choose 'p_inter_area'*'section_width'*'section_width' neurons
        from each sub-network. These neurons can form inter-area (bottom-up and top-down) connections.
        This process just make sure the locations of pre-synaptic inter-area projection neurons are evenly distributed in the
        network. It does not make too much difference on network dynamics compared with just randomly choose 'p_inter_area'*'section_width'*'section_width' neurons
        from the whole network.

        Parameters
        ----------
        p_inter_area : 0~1
            proportion of excitatory neurons that can form inter-area connections.
        N_x_all : 
            width (in number of neurons) of network (x axis).
        N_y_all : TYPE
            width (in number of neurons) of network (y axis).
        section_width : 
            width (in number of neurons) of sub-network

        Returns
        -------
        inter_neuron : 1-D array
            index of pre-synaptic inter-area projection neurons

        '''
        #section_width = 2; N_x_all = 63; N_y_all = 63;
        x_sec = np.arange(0, N_x_all, section_width)
        y_sec = np.arange(0, N_y_all, section_width)
        x_sec[-1] = N_x_all
        y_sec[-1] = N_y_all
        
        lattice_bool = np.zeros([N_y_all, N_x_all],bool)
        #p_inter_area = 1/4
        for j in range(len(y_sec)-1):
            for i in range(len(x_sec)-1):
                section_size = (x_sec[i+1] - x_sec[i])*(y_sec[j+1] - y_sec[j])
                select_num = section_size*p_inter_area
                p = 1 - (select_num - int(select_num))
                if np.random.rand() < p:
                    select_num = int(select_num)
                else:
                    select_num = int(select_num) + 1
                tmp = np.zeros(section_size, bool)
                tmp[np.random.choice(np.arange(section_size),select_num,replace=False)] = True
                lattice_bool[y_sec[j]:y_sec[j+1], x_sec[i]:x_sec[i+1]] = tmp.reshape(y_sec[j+1] - y_sec[j], x_sec[i+1] - x_sec[i])
        
        inter_neuron = np.arange(N_y_all*N_x_all)[lattice_bool.reshape(-1)]
        return inter_neuron

#%%

    
  