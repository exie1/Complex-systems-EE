#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:12:50 2021

@author: shni2598
"""

from coordination import lattice_dist
import numpy as np
#%%

def get_LFP(lattice, LFP_elec, width = 64, LFP_sigma = 8, LFP_effect_range = 2.5):
    #w = np.zeros((len(LFP_elec),N))
    w, i, j  = [], [], []
    neuron_id = np.arange(lattice.shape[0], dtype = int)
    #sigma = 8.
    for pos_i, pos in enumerate(LFP_elec):
        
        dist = lattice_dist(lattice,width,pos)
        i_ = neuron_id[dist < (LFP_sigma*LFP_effect_range)]
        
        w_ = np.exp(-((dist)**2/(2*LFP_sigma**2)))[i_]
        
        j_ = (np.ones(i_.shape)*pos_i).astype(int)
        
        w.append(w_)
        i.append(i_)
        j.append(j_)
    
    w = np.concatenate(w)
    i = np.concatenate(i)
    j = np.concatenate(j)
        
    return i, j, w

#%%
LFP_recordneuron = '''
lfp : amp
'''
LFP_syn = '''
w : 1
lfp_post = (I_exi_pre + I_inh_pre)*w*(-1) : amp (summed)
'''
#%%

# lfp_post = (I_e_pre + I_i_pre + I_extnl_pre)*w*(-1) : amp (summed)

# lfpe_w =  LFP_w(ijwd.lattice_ext, LFP_elec, ijwd.Ne)
# #%%
# #plt.matshow(w[8,:].reshape(63,63))
# LFP_elec = np.array([[0,0],[-32,-32]])

# LFP_elec = np.meshgrid(np.linspace(-63/4*3/2, 63/4*3/2, 4),np.linspace(-63/4*3/2, 63/4*3/2, 4)[::-1])
# LFP_elec = np.concatenate((LFP_elec[0].reshape(-1,1), LFP_elec[1].reshape(-1,1)),1)
# LFP_elec[5,:] = np.array([-1,1]) 

# group_LFP_record = NeuronGroup(len(LFP_elec), model = LFP_recordneuron)
# syn_lfp = Synapses(group_e, group_LFP_record, model = LFP_syn)
# syn_lfp.connect()
# syn_lfp.w[:] = lfpe_w[:]
