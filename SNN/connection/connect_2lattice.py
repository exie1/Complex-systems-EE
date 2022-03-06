# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:22:04 2019

@author: nishe
"""

import numpy as np
from connection import coordination 
#%%
def fix_indegree(lattice_src, lattice_trg, degree_in_trg, tau_d, width, src_equal_trg = False, self_cnt = False):
#%%
    N_src = len(lattice_src)
    N_trg = len(lattice_trg)
    
    j = np.zeros(sum(degree_in_trg), dtype=int)
    i = np.zeros(sum(degree_in_trg), dtype=int)
    dist_ij = np.zeros(sum(degree_in_trg))
    
#    randN = np.random.permutation(N_trg)
    
    pre_ind = 0
    for neuron in range(N_trg):
        
        pre_dist = coordination.lattice_dist(lattice_src, width, lattice_trg[neuron])
        dist_factor = np.exp(-pre_dist/tau_d)
        
        if src_equal_trg and not(self_cnt):
            dist_factor[neuron] = np.nan
        
        choose_src = np.argsort(np.random.rand(N_src)/dist_factor)[:degree_in_trg[neuron]]
        
        j[pre_ind:pre_ind + degree_in_trg[neuron]] = neuron
        i[pre_ind:pre_ind + degree_in_trg[neuron]] = choose_src
        dist_ij[pre_ind:pre_ind + degree_in_trg[neuron]] = pre_dist[choose_src]
        
        pre_ind += degree_in_trg[neuron]
        
    return i, j, dist_ij
#%%       
def fix_outdegree(lattice_src, lattice_trg, degree_out_src, tau_d, width, src_equal_trg = False, self_cnt = False):
           
    N_src = len(lattice_src)
    N_trg = len(lattice_trg)
    
    j = np.zeros(sum(degree_out_src), dtype=int)
    i = np.zeros(sum(degree_out_src), dtype=int)
    dist_ij = np.zeros(sum(degree_out_src))
    
#    randN = np.random.permutation(N_src)
    
    pre_ind = 0
    for neuron in range(N_src):
        
        pre_dist = coordination.lattice_dist(lattice_trg, width, lattice_src[neuron])
        dist_factor = np.exp(-pre_dist/tau_d)
        
        if src_equal_trg and not(self_cnt):
            dist_factor[neuron] = np.nan
        
        choose_trg = np.argsort(np.random.rand(N_trg)/dist_factor)[:degree_out_src[neuron]]
        
        j[pre_ind:pre_ind + degree_out_src[neuron]] = choose_trg
        i[pre_ind:pre_ind + degree_out_src[neuron]] = neuron
        dist_ij[pre_ind:pre_ind + degree_out_src[neuron]] = pre_dist[choose_trg]
        
        pre_ind += degree_out_src[neuron]
        
    return i, j, dist_ij

#%% generate synapses(connections) based on probability that decays exponentially as distance between neurons(soma) increases 
# lattice_src: source group neuron coordination
# lattice_trg: target group neruon coordination
# source_neuron: index of neurons in source group which form synapses \
# peak_p: peak probability
# tau_d: spatial constant of exponential decay probability
# src_equal_trg: if source and target groups are the same
# if self connection (a neuron form synapse with itself) permitted    
    
def expo_decay(lattice_src, lattice_trg, source_neuron, width, periodic_boundary, interarea_dist, peak_p, tau_d, src_equal_trg = False, self_cnt = False):
           
    #N_src = len(lattice_src)
    N_trg = len(lattice_trg)
    
    #j = np.zeros(sum(degree_out_src), dtype=int)
    #i = np.zeros(sum(degree_out_src), dtype=int)
    i = np.array([], int)
    j = np.array([], int)
    dist_ij = np.array([])
    
#    randN = np.random.permutation(N_src)
    
    pre_ind = 0
    for neuron in source_neuron:
        
        if periodic_boundary:
            all_dist = coordination.lattice_dist(lattice_trg, width, lattice_src[neuron])
        else:
            all_dist = coordination.lattice_dist_nonperiodic(lattice_trg, lattice_src[neuron])
        
        all_dist = np.sqrt(all_dist**2 + interarea_dist**2)
        prob = peak_p * np.exp(-all_dist/tau_d)
        
        if src_equal_trg and not(self_cnt):
            prob[neuron] = -1 # make self-connection impossible #np.nan
        
        choose_trg = np.arange(N_trg, dtype=int)[np.random.rand(N_trg) < prob]
        
#        choose_trg = np.argsort(np.random.rand(N_trg)/dist_factor)[:degree_out_src[neuron]]
        
#        j[pre_ind:pre_ind + degree_out_src[neuron]] = choose_trg
#        i[pre_ind:pre_ind + degree_out_src[neuron]] = neuron
#        dist_ij[pre_ind:pre_ind + degree_out_src[neuron]] = all_dist[choose_trg]
        
#        j[pre_ind:pre_ind + len(choose_trg)] = choose_trg
#        i[pre_ind:pre_ind + len(choose_trg)] = neuron
        #dist_ij[pre_ind:pre_ind + len(choose_trg)] = all_dist[choose_trg]
        
        j = np.concatenate((j, choose_trg))
        i = np.concatenate((i, np.ones(len(choose_trg),int)*neuron))
        dist_ij = np.concatenate((dist_ij, all_dist[choose_trg]))
        
        pre_ind += len(choose_trg)
        
    return i, j, dist_ij

def gaussian_decay(lattice_src, lattice_trg, source_neuron, width, periodic_boundary, interarea_dist, peak_p, sig_d, src_equal_trg = False, self_cnt = False):
           
        #N_src = len(lattice_src)
    N_trg = len(lattice_trg)
    
    #j = np.zeros(sum(degree_out_src), dtype=int)
    #i = np.zeros(sum(degree_out_src), dtype=int)
    i = np.array([], int)
    j = np.array([], int)
    dist_ij = np.array([])
    
#    randN = np.random.permutation(N_src)
    
    #pre_ind = 0
    for neuron in source_neuron:
        
        if periodic_boundary:
            all_dist = coordination.lattice_dist(lattice_trg, width, lattice_src[neuron])
        else:
            all_dist = coordination.lattice_dist_nonperiodic(lattice_trg, lattice_src[neuron])
        
        all_dist = np.sqrt(all_dist**2 + interarea_dist**2)
        prob = peak_p * np.exp(-(all_dist/sig_d)**2/2)
        
        if src_equal_trg and not(self_cnt):
            prob[neuron] = -1 # make self-connection impossible #np.nan
        
        choose_trg = np.arange(N_trg, dtype=int)[np.random.rand(N_trg) < prob]
                
        j = np.concatenate((j, choose_trg))
        i = np.concatenate((i, np.ones(len(choose_trg),int)*neuron))
        dist_ij = np.concatenate((dist_ij, all_dist[choose_trg]))
        
        #pre_ind += len(choose_trg)
        
    return i, j, dist_ij



    


