#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:23:17 2021

@author: shni2598
"""


from brian2.only import *

import connection as cn
#%%
class one_area:
    
    def __init__(self):
        
        self.neuron_model_e = cn.model_neu_syn_AD.neuron_e_AD_t
        self.neuron_model_i = cn.model_neu_syn_AD.neuron_i_AD_t
        self.synapse_model_e = cn.model_neu_syn_AD.synapse_e_AD
        self.synapse_model_i = cn.model_neu_syn_AD.synapse_i_AD 
        
        
    def build(self, ijwd1):
        
        group_e_1 = NeuronGroup(ijwd1.Ne, model=self.neuron_model_e,
                     threshold='v>v_threshold', method='euler',
                     reset='''v = v_reset
                              g_k += delta_gk''', refractory='(t-lastspike)<t_ref')

        group_i_1 = NeuronGroup(ijwd1.Ni, model=self.neuron_model_i,
                             threshold='v>v_threshold', method='euler',
                             reset='v = v_reset', refractory='(t-lastspike)<t_ref')
        
        syn_ee_1 = Synapses(group_e_1, group_e_1, model=self.synapse_model_e, 
                            on_pre='''x_E_post += w''')
        syn_ei_1 = Synapses(group_e_1, group_i_1, model=self.synapse_model_e, 
                            on_pre='''x_E_post += w''')
        syn_ie_1 = Synapses(group_i_1, group_e_1, model=self.synapse_model_i, 
                            on_pre='''x_I_post += w''')
        syn_ii_1 = Synapses(group_i_1, group_i_1, model=self.synapse_model_i,
                            on_pre='''x_I_post += w''')

        syn_ee_1.connect(i=ijwd1.i_ee, j=ijwd1.j_ee)
        syn_ei_1.connect(i=ijwd1.i_ei, j=ijwd1.j_ei)
        syn_ie_1.connect(i=ijwd1.i_ie, j=ijwd1.j_ie)
        syn_ii_1.connect(i=ijwd1.i_ii, j=ijwd1.j_ii)
        
        
        #tau_s_di_try = tau_s_di_
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
        
        
        return group_e_1, group_i_1, syn_ee_1, syn_ei_1, syn_ie_1, syn_ii_1
