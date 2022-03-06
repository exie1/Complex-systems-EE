# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:47:50 2020

@author: nishe
"""

import brian2.numpy_ as np
import matplotlib.pyplot as plt
import connection as cn
#import warnings
from scipy import sparse
from brian2.only import *
import time
import mydata
import firing_rate_analysis
import os
import datetime

import expo_connection
#%%
prefs.codegen.target = 'numpy'

#%%
neuronmodel_e = cn.model_neu_syn_AD.neuron_e_AD
#neuronmodel_i = cn.model_neu_syn_AD.neuron_i_AD

synapse_e = cn.model_neu_syn_AD.synapse_e_AD
#synapse_i = cn.model_neu_syn_AD.synapse_i_AD_SP
#%%
Ne = 1
start_scope()

group_e_1 =NeuronGroup(Ne, model=neuronmodel_e,
                     threshold='v>v_threshold', method='euler',
                     reset='''v = v_reset
                              g_k += delta_gk''', refractory='(t-lastspike)<t_ref')

#syn_ee_1 = Synapses(group_e_1, group_e_1, model=synapse_e, 
#                  on_pre='''x_E_post += w
#                            ''')

#syn_ee_1 = Synapses(group_e_1, group_e_1, model=synapse_e, 
#                  on_pre='''x_E_post += w*u*L
#                            L -= u*L''')

#indices = np.zeros(50)
#times =np.arange(5, 51, 5)*ms
#inpt = SpikeGeneratorGroup(1, indices, times)
#%%

#syn_ee_1.connect(i=[0], j=[0])

#tau_s_di_try = tau_s_di_
#syn_ee_1.w = 4 * 5.8 * nS #tau_s_de_
#u_base = 0.3
#syn_ee_1.u = u_base; syn_ee_1.L = 1
#syn_ei_1.u = u_base; syn_ee_1.L = 1
#syn_ii_1.u = u_base; syn_ee_1.L = 1
#syn_ie_1.u = u_base; syn_ee_1.L = 1
#tau_f = 1500*ms; tau_d = 200*ms

#def set_delay(syn, delay_up):
#    #n = len(syn)
#    syn.delay = delay_up*ms
#    #syn.down.delay = (delay_up + 1)*ms
#    
#    return syn 

#generate_d_dist()
#generate_d_rand()
#def generate_d_rand(delay,len_i_ee,len_i_ie,len_i_ei,len_i_ii):
#        
#    d_ee = np.random.uniform(0, delay, len_i_ee)    
#    d_ie = np.random.uniform(0, delay, len_i_ie)
#    d_ei = np.random.uniform(0, delay, len_i_ei)
#    d_ii = np.random.uniform(0, delay, len_i_ii)
#    
#    return d_ee, d_ie, d_ei, d_ii

#d_ee, d_ie, d_ei, d_ii = generate_d_rand(4,len(i_ee),len(i_ie),len(i_ei),len(i_ii))
#syn_ee_1 = set_delay(syn_ee_1, ijwd.d_ee)
#syn_ie_1 = set_delay(syn_ie_1, ijwd.d_ie)
#syn_ei_1 = set_delay(syn_ei_1, ijwd.d_ei)
#syn_ii_1 = set_delay(syn_ii_1, ijwd.d_ii)

#syn_pois_e = set_delay(syn_pois_e)
#syn_pois_i = set_delay(syn_pois_i)

tau_s_de_ = 5.8; tau_s_di_ = 6.5
delta_gk_ = 10


group_e_1.tau_s_de = tau_s_de_*ms; 
group_e_1.tau_s_di = tau_s_di_*ms
group_e_1.tau_s_re = group_e_1.tau_s_ri = 1*ms

group_e_1.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
group_e_1.tau_s_re_inter = 1*ms
group_e_1.tau_s_de_extnl = tau_s_de_*ms #5.0*ms
group_e_1.tau_s_re_extnl = 1*ms

#group_i_1.tau_s_de = tau_s_de_*ms
#group_i_1.tau_s_di = tau_s_di_*ms
#group_i_1.tau_s_re = group_i_1.tau_s_ri = 1*ms
#
#group_i_1.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
#group_i_1.tau_s_re_inter = 1*ms
#group_i_1.tau_s_de_extnl = tau_s_de_*ms #5.0*ms
#group_i_1.tau_s_re_extnl = 1*ms

#syn_pois_e.effect = 0
#syn_pois_i.effect = 0
#group_e_v2.v = np.random.random(Ne)*35*mV-85*mV
#group_i_v2.v = np.random.random(Ni)*35*mV-85*mV
#seed(1000)
group_e_1.v = -70*mV#np.random.random(ijwd.Ne)*35*mV-85*mV
#group_i_1.v = np.random.random(ijwd.Ni)*35*mV-85*mV
delta_gk_ = 10
group_e_1.delta_gk = delta_gk_*nS
group_e_1.tau_k = 80*ms
#group_e_v2.v = np.random.random(Ne)*10*mV-60*mV
#group_i_v2.v = np.random.random(Ni)*10*mV-60*mV
group_e_1.I_extnl_crt = 0.7*nA #0.51*nA
#group_i_1.I_extnl_crt = 0*nA #0.60*nA

#%%
spk_e = SpikeMonitor(group_e_1, record = True)
s_moni = StateMonitor(group_e_1, ('v','I_exi'), record = True)
#%%
net = Network(collect())
net.store('state1')
#%%
#change_ie(4.4)
#syn_ie_1.w = w_ie*usiemens

#print('ie_w: %fnsiemens' %(syn_ie_1.w[0]/nsiemens))
#Ne = 63*63; Ni = 1000;
C = 0.25*nF # capacitance
g_l = 16.7*nS # leak capacitance
v_l = -70*mV # leak voltage
v_threshold = -50*mV
v_reset = -70*mV
v_rev_I = -80*mV
v_rev_E = 0*mV
v_k = -85*mV
#tau_k = 80*ms# 80*ms
#delta_gk = 10*nS #10*nS
t_ref = 4*ms # refractory period
delta_gk_ = 0
group_e_1.delta_gk = delta_gk_*nS

tic = time.perf_counter()
#seed(10)
simu_time1 = 500*ms#2000*ms
#simu_time2 = 2000*ms
#simu_time2 = 2000*ms#8000*ms
#simu_time3 = 1000*ms
#simu_time_tot = 30000*ms
#group_input.active = False
#extnl_e.rates = bkg_rate2e
net.run(simu_time1, profile=False) #,namespace={'tau_k': 80*ms}
#extnl_e.rates = bkg_rate2e + stim_rate
#group_e_1.delta_gk[chg_adapt_neuron] = new_delta_gk_*nS
#net.run(simu_time2, profile=False) #,namespace={'tau_k': 80*ms}
#extnl_e.rates = bkg_rate2e
#net.run(simu_time3, profile=False) #,namespace={'tau_k': 80*ms}
#group_input.active = True
#group_e_1.delta_gk[chg_adapt_neuron] = 2*nS; group_e_1.tau_k[chg_adapt_neuron] = 40*ms
#net.run(simu_time2, report = 'text', profile=False) #,namespace={'tau_k': 80*ms}
print('total time elapsed:',time.perf_counter() - tic)
#%%
plt.figure()
plt.plot(s_moni.t/ms, s_moni.v[0]/mV)
#%%
plt.figure()
plt.plot(spk_e.t/ms, spk_e.i, '|')
#%%
spk_e.i.shape

#%%
net.restore('state1')


