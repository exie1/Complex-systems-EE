#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 11:45:34 2021

@author: shni2598
"""


from multiprocessing import connection
from connection import pre_process_sc
from connection import build_one_area
from connection import coordination
from connection import poisson_stimuli
from connection import model_neu_syn_AD
import brian2.numpy_ as np
from brian2.only import *
import time
import os
import datetime
import sys
import pickle
import random

def find_w_e(w_i, num_i, num_e, ie_ratio):
    return (w_i*num_i)/num_e/ie_ratio
               
def find_w_i(w_e, num_e, num_i, ie_ratio):
    return (w_e*num_e)*ie_ratio/num_i

# def set_stimulus(lattice,centre,radius): # Find the set of lattice indices for a circular stimulus
#     x = lattice[:,0]
#     y = lattice[:,1]
#     r = ((x-centre[0])**2 + (y-centre[1])**2)**0.5
#     stim_index_base = np.argwhere(r<=radius)
#     stim_index = [int(i) for i in stim_index_base]
#     return stim_index


synapse_model_e = model_neu_syn_AD.synapse_e_AD


ijwd1 = pre_process_sc.get_ijwd()
ijwd1.Ne = 64*64; 
ijwd1.Ni = 32*32 
ijwd1.width = 64 

pos = -25
ijwd1.w_extnl = 25
ijwd1.stim_pos = pos

ijwd1.decay_p_ee = 7 # e to e
ijwd1.decay_p_ei = 9 # e to i
ijwd1.decay_p_ie = 19 # i to e
ijwd1.decay_p_ii = 19 # i to i
ijwd1.delay = [0.5,2.5] # ms; spike delay

'''average in-degree'''
num_ee = 240; num_ei = 400; num_ie = 150; num_ii = 230

ijwd1.mean_SynNumIn_ee = num_ee # e to e
ijwd1.mean_SynNumIn_ei = num_ei # e to i
ijwd1.mean_SynNumIn_ie = num_ie # i to e
ijwd1.mean_SynNumIn_ii = num_ii # i to i

'''synaptic weight'''
ie_r_e = 3.2786 # 2.76*6.5/5.8*1.06   ie ratio for exitatory neurons
ie_r_i = 2.9104 # 2.450*6.5/5.8*1.06  ie ratio for inhibitory neurons
w_ie = 115 # nsiemens; synaptic weight i to e
w_ii = 140 # nsiemens; i to i
ijwd1.w_ee_mean = find_w_e(w_ie, num_ie, num_ee, ie_r_e) # nsiemens; e to e ~15/20
ijwd1.w_ei_mean = find_w_e(w_ii, num_ii, num_ei, ie_r_i) # nsiemens; e to i
ijwd1.w_ie_mean = w_ie # nsiemens; i to e
ijwd1.w_ii_mean = w_ii # nsiemens; i to i

'''generate connectivity'''
ijwd1.generate_ijw()
ijwd1.generate_d_dist()


# input_list = random.sample(list(ijwd1.w_ee),int(len(ijwd1.stim_index)/2))
# ijwd1.w_inp = input_list + input_list

param_a1 = {**ijwd1.__dict__}

del param_a1['i_ee'], param_a1['j_ee'], param_a1['w_ee'], param_a1['d_ee'], param_a1['dist_ee'] 
del param_a1['i_ei'], param_a1['j_ei'], param_a1['w_ei'], param_a1['d_ei'], param_a1['dist_ei']
del param_a1['i_ie'], param_a1['j_ie'], param_a1['w_ie'], param_a1['d_ie'], param_a1['dist_ie'] 
del param_a1['i_ii'], param_a1['j_ii'], param_a1['w_ii'], param_a1['d_ii'], param_a1['dist_ii']


start_scope()


onearea_net = build_one_area.one_area()

group_e_1, group_i_1, syn_ee_1, syn_ei_1, syn_ie_1, syn_ii_1, posi_stim_e1, syn_extnl_e1 = onearea_net.build(ijwd1)



delta_gk_ = 16 # nsiemens; adaptation potassium conductance change after spike generation 
tau_k_ = 60 # ms; decay time constant of adaptation potassium conductance

tau_s_di_ = 4.4 # ms; decay time constant of inhibitory current
tau_s_de_ = 5.  # ms; decay time constant of excitatory current
tau_s_r_ = 1 # ms; rising time constant of both excitatory and inhibitory current

I_extnl_crt2e = 0.51 # nAmp; back|ground current to excitatory neurons
I_extnl_crt2i = 0.60 # nAmp; background current to inhibitory neurons

C = 0.25*nF # nF;  membrane capacitance
g_l = 16.7*nS # nS; leak capacitance
v_l = -70*mV # mV; leak voltage
v_threshold = -50*mV # mV; spike threshold
v_reset = -70*mV # mV;  reset voltagge
v_rev_I = -80*mV # mV;  reverse voltage for inhibitory synaptic current
v_rev_E = 0*mV # mV;  reverse voltage for exitatory synaptic current
v_k = -85*mV # mV;  reverse voltage for adaptation potassium current
t_ref = 5*ms # ms;  refractory period


group_e_1.tau_s_de = tau_s_de_*ms; 
group_e_1.tau_s_di = tau_s_di_*ms
group_e_1.tau_s_re = group_e_1.tau_s_ri = tau_s_r_*ms

group_e_1.tau_s_de_inter = tau_s_de_*ms # decay time constant of exci current induced by inter-area spikes (spikes from other areas)
group_e_1.tau_s_re_inter = 1*ms # rising time constant of exci current induced by inter-area spikes (spikes from other areas)
group_e_1.tau_s_de_extnl = 5.0*ms # decay time constant of exci current induced by external spikes
group_e_1.tau_s_re_extnl = 1*ms # rising time constant of exci current induced by external spikes

group_i_1.tau_s_de = tau_s_de_*ms
group_i_1.tau_s_di = tau_s_di_*ms
group_i_1.tau_s_re = group_i_1.tau_s_ri = tau_s_r_*ms

group_i_1.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
group_i_1.tau_s_re_inter = 1*ms
group_i_1.tau_s_de_extnl = 5.0*ms #5.0*ms
group_i_1.tau_s_re_extnl = 1*ms

group_e_1.v = np.random.random(ijwd1.Ne)*35*mV-85*mV
group_i_1.v = np.random.random(ijwd1.Ni)*35*mV-85*mV
group_e_1.delta_bkg = delta_gk_*nS
group_e_1.tau_k = tau_k_*ms

group_e_1.I_extnl_crt = I_extnl_crt2e*nA # 0.25 0.51*nA
group_i_1.I_extnl_crt = I_extnl_crt2i*nA # 0.25 0.60*nA


spk_e_1 = SpikeMonitor(group_e_1, record = True)
spk_i_1 = SpikeMonitor(group_i_1, record = True)


# #---------------------MOVING STIMULUS------------------------------
# pos_dict = {'pos':-25}
# @network_operation(dt = 20*ms)
# def change_pos():
#     pos_dict['pos'] += 1
#     posi_stim_e1.stim_1 = poisson_stimuli.input_spkrate(maxrate = [500], \
#         sig=[4], position=[[pos_dict['pos'],0]])*Hz
# #-----------------------------------------------------------------

net = Network(collect())
net.store('state1') 

#%%
tic = time.perf_counter()
simu_time_tot = 1000*ms # simulation time

simu_time1 = simu_time_tot 
net.run(simu_time1, profile=False) 

print('total time elapsed:',time.perf_counter() - tic, 'seconds')

#%%
'''save data to hard disk'''
spk_tstep_e1 = np.round(spk_e_1.t/(0.1*ms)).astype(int)
spk_tstep_i1 = np.round(spk_i_1.t/(0.1*ms)).astype(int)

param_all = {'delta_gk':delta_gk_,
         #'new_delta_gk':new_delta_gk_,
         'tau_k': 60,
         #'new_tau_k':40,
         'tau_s_di':tau_s_di_,
         'tau_s_de':tau_s_de_,
         'tau_s_r':tau_s_r_,
         #'scale_d_p_i':scale_d_p_i,
         'num_ee':num_ee,
         'num_ei':num_ei,
         'num_ii':num_ii,
         'num_ie':num_ie,
         #'ie_ratio':ie_ratio_,
         #'mean_J_ee': ijwd1.mean_J_ee,
         #'chg_adapt_range':6, 
         #'p_ee':p_ee,
         #'simutime':int(round(simu_time_tot/ms)),
         #'chg_adapt_time': simu_time1/ms,
         #'chg_adapt_range': chg_adapt_range,
         # 'chg_adapt_loca': chg_adapt_loca,
         #'chg_adapt_neuron': chg_adapt_neuron,
         #'scale_ee_1': scale_ee_1,
         #'scale_ei_1': scale_ei_1,
         #'scale_ie_1': scale_ie_1,
         #'scale_ii_1': scale_ii_1,
         # 'ie_r_e': ie_r_e,
         # 'ie_r_e1':ie_r_e1,   
         # #'ie_r_e2':ie_r_e2,
         # 'ie_r_i': ie_r_i,
         # 'ie_r_i1': ie_r_i1,
         't_ref': t_ref/ms}

now = datetime.datetime.now()
data = {'datetime':now.strftime("%Y-%m-%d %H:%M:%S"), 'dt':0.1, #'loop_num':loop_num, 
        'data_dir': os.getcwd(),
        'param':param_all,
        'a1':{'param':param_a1,
              #'LFP':{'lfp1':lfp_moni.lfp1/nA, 'lfp2':lfp_moni.lfp2/nA, 'lfp3':lfp_moni.lfp3/nA},
              'ge':{'i':spk_e_1.i[:],'t':spk_tstep_e1},
              'gi':{'i':spk_i_1.i[:],'t':spk_tstep_i1}}}

print(data['a1']['ge'])
# os.chdir('C:/Users/Evan Xie/Desktop/SBstuff/model_code')
# filename = 'one_stim.pickle'
# with open(filename, 'wb') as handle:
#     pickle.dump(data, handle)
# handle.close()
# print(filename)


#-------------OPTIONAL PLOTTING----------------
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(data['a1']['ge']['t'],data['a1']['ge']['i'],'.',markersize=1)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index')
plt.title('Spike times per neuron index')
plt.show()



# %%
