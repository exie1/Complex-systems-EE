#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:43:27 2020

@author: shni2598
"""

from brian2.only import *

import connection as cn
#%%
class two_areas:
    
    def __init__(self):
        
        self.neuron_model_e = cn.model_neu_syn_AD.neuron_e_AD
        self.neuron_model_e_adapt_tv = cn.model_neu_syn_AD.neuron_e_AD_adapt_tv
        self.neuron_model_i = cn.model_neu_syn_AD.neuron_i_AD
        self.synapse_model_e = cn.model_neu_syn_AD.synapse_e_AD
        self.synapse_model_i = cn.model_neu_syn_AD.synapse_i_AD 
        
        
    def build(self, ijwd1, ijwd2, ijwd_inter, adapt_modu_1 = False, adapt_modu_2 = False):
        
        if not adapt_modu_1:
            group_e_1 = NeuronGroup(ijwd1.Ne, model=self.neuron_model_e,
                         threshold='v>v_threshold', method='euler',
                         reset='''v = v_reset
                                  g_k += delta_gk''', refractory='(t-lastspike)<t_ref')
        else:
            group_e_1 = NeuronGroup(ijwd1.Ne, model=self.neuron_model_e_adapt_tv,
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
        
        if not adapt_modu_2:
            group_e_2 = NeuronGroup(ijwd2.Ne, model=self.neuron_model_e,
                         threshold='v>v_threshold', method='euler',
                         reset='''v = v_reset
                                  g_k += delta_gk''', refractory='(t-lastspike)<t_ref')
        else:
            group_e_2 = NeuronGroup(ijwd2.Ne, model=self.neuron_model_e_adapt_tv,
                         threshold='v>v_threshold', method='euler',
                         reset='''v = v_reset
                                  g_k += delta_gk''', refractory='(t-lastspike)<t_ref')            
            

        group_i_2 = NeuronGroup(ijwd2.Ni, model=self.neuron_model_i,
                             threshold='v>v_threshold', method='euler',
                             reset='v = v_reset', refractory='(t-lastspike)<t_ref')
        
        syn_ee_2 = Synapses(group_e_2, group_e_2, model=self.synapse_model_e, 
                            on_pre='''x_E_post += w''')
        syn_ei_2 = Synapses(group_e_2, group_i_2, model=self.synapse_model_e, 
                            on_pre='''x_E_post += w''')
        syn_ie_2 = Synapses(group_i_2, group_e_2, model=self.synapse_model_i, 
                            on_pre='''x_I_post += w''')
        syn_ii_2 = Synapses(group_i_2, group_i_2, model=self.synapse_model_i, 
                            on_pre='''x_I_post += w''')
        
        syn_ee_2.connect(i=ijwd2.i_ee, j=ijwd2.j_ee)
        syn_ei_2.connect(i=ijwd2.i_ei, j=ijwd2.j_ei)
        syn_ie_2.connect(i=ijwd2.i_ie, j=ijwd2.j_ie)
        syn_ii_2.connect(i=ijwd2.i_ii, j=ijwd2.j_ii)
        
        
        #tau_s_di_try = tau_s_di_
        syn_ee_2.w = ijwd2.w_ee*nsiemens#/tau_s_r_ * 5.8 #tau_s_de_
        syn_ei_2.w = ijwd2.w_ei*nsiemens#/tau_s_r_ * 5.8 #tau_s_de_ #5*nS
        syn_ii_2.w = ijwd2.w_ii*nsiemens#/tau_s_r_ * 6.5 #tau_s_di_#25*nS
        syn_ie_2.w = ijwd2.w_ie*nsiemens#/tau_s_r_ * 6.5 #tau_s_di_#
        
        
        #d_ee, d_ie, d_ei, d_ii = generate_d_rand(4,len(i_ee),len(i_ie),len(i_ei),len(i_ii))
        syn_ee_2 = set_delay(syn_ee_2, ijwd2.d_ee)
        syn_ie_2 = set_delay(syn_ie_2, ijwd2.d_ie)
        syn_ei_2 = set_delay(syn_ei_2, ijwd2.d_ei)
        syn_ii_2 = set_delay(syn_ii_2, ijwd2.d_ii)


        syn_e1e2 = Synapses(group_e_1, group_e_2, model=self.synapse_model_e, 
                            on_pre='''x_E_inter_post += w''')
        syn_e1i2 = Synapses(group_e_1, group_i_2, model=self.synapse_model_e, 
                            on_pre='''x_E_inter_post += w''')
        syn_e2e1 = Synapses(group_e_2, group_e_1, model=self.synapse_model_e, 
                            on_pre='''x_E_inter_post += w''')
        syn_e2i1 = Synapses(group_e_2, group_i_1, model=self.synapse_model_e, 
                            on_pre='''x_E_inter_post += w''')
        
        syn_e1e2.connect(i=ijwd_inter.i_e1_e2, j=ijwd_inter.j_e1_e2)
        syn_e1i2.connect(i=ijwd_inter.i_e1_i2, j=ijwd_inter.j_e1_i2)
        syn_e2e1.connect(i=ijwd_inter.i_e2_e1, j=ijwd_inter.j_e2_e1)
        syn_e2i1.connect(i=ijwd_inter.i_e2_i1, j=ijwd_inter.j_e2_i1)
        
        syn_e1e2.w = ijwd_inter.w_e1_e2*nS#*5*nS
        syn_e1i2.w = ijwd_inter.w_e1_i2*nS#*5*nS
        syn_e2e1.w = ijwd_inter.w_e2_e1*nS#*5*nS
        syn_e2i1.w = ijwd_inter.w_e2_i1*nS#*5*nS
        
        syn_e1e2.delay = ijwd_inter.d_e1_e2*ms
        syn_e1i2.delay = ijwd_inter.d_e1_i2*ms
        syn_e2e1.delay = ijwd_inter.d_e2_e1*ms
        syn_e2i1.delay = ijwd_inter.d_e2_i1*ms
        
        return group_e_1, group_i_1, syn_ee_1, syn_ei_1, syn_ie_1, syn_ii_1,\
                group_e_2, group_i_2, syn_ee_2, syn_ei_2, syn_ie_2, syn_ii_2,\
                syn_e1e2, syn_e1i2, syn_e2e1, syn_e2i1
 
#%%

        



        