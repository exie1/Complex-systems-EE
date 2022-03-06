#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:12:18 2020

@author: shni2598
"""

from re import T
import numpy as np
from brian2.only import *
import connection as cn
from connection import poisson_stimuli
#%%
class one_area:
    
    def __init__(self):
        
        self.neuron_model_e = cn.model_neu_syn_AD.neuron_e_AD
        self.neuron_model_i = cn.model_neu_syn_AD.neuron_i_AD
        self.synapse_model_e = cn.model_neu_syn_AD.synapse_e_AD
        self.synapse_model_i = cn.model_neu_syn_AD.synapse_i_AD 
        
        
    def build(self, ijwd1):

        # group_stim = PoissonGroup(len(ijwd1.stim_index),500*Hz)

        group_e_1 = NeuronGroup(ijwd1.Ne, model=self.neuron_model_e,
                     threshold='v>v_threshold', method='euler',
                     reset='''v = v_reset
                              g_k += delta_bkg + delta_stim''', refractory='(t-lastspike)<t_ref')

        group_i_1 = NeuronGroup(ijwd1.Ni, model=self.neuron_model_i,
                             threshold='v>v_threshold', method='euler',
                             reset='v = v_reset', refractory='(t-lastspike)<t_ref')
        
        # syn_stim = Synapses(group_stim, group_e_1, model=self.synapse_model_e, 
        #                     on_pre='''x_E_post += w''')
        syn_ee_1 = Synapses(group_e_1, group_e_1, model=self.synapse_model_e, 
                            on_pre='''x_E_post += w''')
        syn_ei_1 = Synapses(group_e_1, group_i_1, model=self.synapse_model_e, 
                            on_pre='''x_E_post += w''')
        syn_ie_1 = Synapses(group_i_1, group_e_1, model=self.synapse_model_i, 
                            on_pre='''x_I_post += w''')
        syn_ii_1 = Synapses(group_i_1, group_i_1, model=self.synapse_model_i,
                            on_pre='''x_I_post += w''')

        # syn_stim.connect(i = list(range(len(ijwd1.stim_index))), j = ijwd1.stim_index)
        syn_ee_1.connect(i=ijwd1.i_ee, j=ijwd1.j_ee)
        syn_ei_1.connect(i=ijwd1.i_ei, j=ijwd1.j_ei)
        syn_ie_1.connect(i=ijwd1.i_ie, j=ijwd1.j_ie)
        syn_ii_1.connect(i=ijwd1.i_ii, j=ijwd1.j_ii)
        
        
        #tau_s_di_try = tau_s_di_
        # syn_stim.w = ijwd1.w_inp*nsiemens
        syn_ee_1.w = ijwd1.w_ee*nsiemens#/tau_s_r_ * 5.8 #tau_s_de_
        syn_ei_1.w = ijwd1.w_ei*nsiemens#/tau_s_r_ * 5.8 #tau_s_de_ #5*nS
        syn_ii_1.w = ijwd1.w_ii*nsiemens#/tau_s_r_ * 6.5 #tau_s_di_#25*nS
        syn_ie_1.w = ijwd1.w_ie*nsiemens#/tau_s_r_ * 6.5 #tau_s_di_#
        
        
        def set_delay(syn, delay_up):
            #n = len(syn)
            syn.delay = delay_up*ms
            #syn.down.delay = (delay_up + 1)*ms
            
            return syn 
        
        #d_ee, d_ie, d_ei, d_ii = generate_d_rand(4,len(i_ee),len(i_ie),len(i_ei),len(i_ii))
        syn_ee_1 = set_delay(syn_ee_1, ijwd1.d_ee)
        syn_ie_1 = set_delay(syn_ie_1, ijwd1.d_ie)
        syn_ei_1 = set_delay(syn_ei_1, ijwd1.d_ei)
        syn_ii_1 = set_delay(syn_ii_1, ijwd1.d_ii)

        posi_stim_e1 = NeuronGroup(ijwd1.Ne,  
                        '''rates =  bkg_rates + stim_1 + stim_2: Hz
                        bkg_rates : Hz
                        stim_1 : Hz
                        stim_2 : Hz
                        ''', threshold='rand()<rates*dt')

        posi_stim_e1.bkg_rates = 0*Hz
        posi_stim_e1.stim_1 = poisson_stimuli.input_spkrate(maxrate = [500], sig=[6], position=[[0,0]])*Hz
        posi_stim_e1.stim_2 = poisson_stimuli.input_spkrate(maxrate = [0], sig=[6], position=[[-32,-32]])*Hz
        group_e_1.delta_stim = poisson_stimuli.input_adaptation(maxdelta=[0,0],sig=[6,6],position=[[0,0],[-32,-32]])*nS
        #posi_stim_e1.stim_2 = psti.input_spkrate(maxrate = [200], sig=[6], position=[[-32, 0]])*Hz

        syn_extnl_e1 = Synapses(posi_stim_e1, group_e_1, model=self.synapse_model_e, on_pre='x_E_extnl_post += w')
        syn_extnl_e1.connect('i==j')
        syn_extnl_e1.w = ijwd1.w_extnl*nS#*tau_s_de_*nS


        
        
        return group_e_1, group_i_1, syn_ee_1, syn_ei_1, syn_ie_1, syn_ii_1, posi_stim_e1, syn_extnl_e1
