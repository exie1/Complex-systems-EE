#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:35:43 2021

@author: shni2598
"""

from connection import pre_process_sc

#%%
def find_w_e(w_i, num_i, num_e, ie_ratio):
    return (w_i*num_i)/num_e/ie_ratio
               
def find_w_i(w_e, num_e, num_i, ie_ratio):
    return (w_e*num_e)*ie_ratio/num_i


#%%

ijwd = pre_process_sc.get_ijwd()
ijwd.Ne = 64*64; # number of exitatory neurons
ijwd.Ni = 32*32 # number of inhibitory neurons
ijwd.width = 64 # width of network (in number of neurons)

'''decay constant of connection probability, which is a exponential function of distance'''
ijwd.decay_p_ee = 7 # e to e
ijwd.decay_p_ei = 9 # e to i
ijwd.decay_p_ie = 19 # i to e
ijwd.decay_p_ii = 19 # i to i
'''spike delay'''
ijwd.delay = [0.5,2.5] # ms; spike delay

'''average in-degree
note that there is a maximum number of in-degree for a given decay constant of connection probability'''
num_ee = 240; num_ei = 400; num_ie = 150; num_ii = 230

ijwd.mean_SynNumIn_ee = num_ee # e to e
ijwd.mean_SynNumIn_ei = num_ei # e to i
ijwd.mean_SynNumIn_ie = num_ie # i to e
ijwd.mean_SynNumIn_ii = num_ii # i to i

'''synaptic weight'''
ie_r_e = 3.2786 # 2.76*6.5/5.8*1.06   ie ratio for exitatory neurons
ie_r_i = 2.9104 # 2.450*6.5/5.8*1.06  ie ratio for inhibitory neurons
w_ie = 115 # nsiemens; synaptic weight i to e
w_ii = 140 # nsiemens; i to i
ijwd.w_ee_mean = find_w_e(w_ie, num_ie, num_ee, ie_r_e) # nsiemens; e to e
ijwd.w_ei_mean = find_w_e(w_ii, num_ii, num_ei, ie_r_i) # nsiemens; e to i
ijwd.w_ie_mean = w_ie # nsiemens; i to e
ijwd.w_ii_mean = w_ii # nsiemens; i to i

'''generate connectivity'''
ijwd.generate_ijw()
ijwd.generate_d_dist()

param = {**ijwd.__dict__}

del param['i_ee'], param['j_ee'], param['w_ee'], param['d_ee'], param['dist_ee'] 
del param['i_ei'], param['j_ei'], param['w_ei'], param['d_ei'], param['dist_ei']
del param['i_ie'], param['j_ie'], param['w_ie'], param['d_ie'], param['dist_ie'] 
del param['i_ii'], param['j_ii'], param['w_ii'], param['d_ii'], param['dist_ii']

#%%
'''other parameters'''
delta_gk = 16 # nsiemens; adaptation potassium conductance change after spike generation 
tau_k_ = 60 # ms; decay time constant of adaptation potassium conductance

tau_s_di_ = 4.4 # ms; decay time constant of inhibitory current
tau_s_de_ = 5.  # ms; decay time constant of excitatory current
tau_s_r_ = 1 # ms; rising time constant of both excitatory and inhibitory current

I_extnl_crt2e = 0.51 # nAmp; background current to excitatory neurons
I_extnl_crt2i = 0.60 # nAmp; background current to inhibitory neurons

C = 0.25 # nF;  membrane capacitance
g_l = 16.7 # nS; leak capacitance
v_l = -70 # mV; leak voltage
v_threshold = -50 # mV; spike threshold
v_reset = -70 # mV;  reset voltagge
v_rev_I = -80 # mV;  reverse voltage for inhibitory synaptic current
v_rev_E = 0 # mV;  reverse voltage for exitatory synaptic current
v_k = -85 # mV;  reverse voltage for adaptation potassium current
t_ref = 5 # ms;  refractory period

#%%

'''neuron and synapse model; can be found in ./connection/model_neu_syn_AD.py'''


neuron_e_AD = '''
dv/dt = (1/C)*(-g_l*(v - v_l) + (-g_k)*(v - v_k) + I_inh + I_exi + I_extnl_spk + I_extnl_crt) : volt (unless refractory)

I_inh = (-g_I)*(v - v_rev_I) : amp
I_exi = (-g_E - g_E_inter)*(v - v_rev_E): amp
I_extnl_spk = (-g_E_extnl)*(v - v_rev_E) : amp

dg_k/dt = -g_k/tau_k :siemens
delta_gk : siemens
tau_k : second

dg_E_inter/dt = -g_E_inter/tau_s_de_inter + x_E_inter/tau_s_de_inter : siemens
dx_E_inter/dt = -x_E_inter/tau_s_re_inter : siemens
tau_s_de_inter : second
tau_s_re_inter : second


dg_E/dt = -g_E/tau_s_de + x_E/tau_s_de : siemens
dx_E/dt = -x_E/tau_s_re : siemens
tau_s_de : second
tau_s_re : second

dg_I/dt = -g_I/tau_s_di + x_I/tau_s_di : siemens
dx_I/dt = -x_I/tau_s_ri : siemens
tau_s_di : second
tau_s_ri : second

dg_E_extnl/dt = -g_E_extnl/tau_s_de_extnl + x_E_extnl/tau_s_de_extnl : siemens
dx_E_extnl/dt = -x_E_extnl/tau_s_re_extnl : siemens
tau_s_de_extnl : second
tau_s_re_extnl : second

I_extnl_crt : amp

'''

neuron_i_AD = '''
dv/dt = (1/C)*(-g_l*(v - v_l) + I_inh + I_exi + I_extnl_spk + I_extnl_crt) : volt (unless refractory)

I_inh = (-g_I)*(v - v_rev_I) : amp
I_exi = (-g_E - g_E_inter)*(v - v_rev_E): amp
I_extnl_spk = (-g_E_extnl)*(v - v_rev_E) : amp

dg_E_inter/dt = -g_E_inter/tau_s_de_inter + x_E_inter/tau_s_de_inter : siemens
dx_E_inter/dt = -x_E_inter/tau_s_re_inter : siemens
tau_s_de_inter : second
tau_s_re_inter : second


dg_E/dt = -g_E/tau_s_de + x_E/tau_s_de : siemens
dx_E/dt = -x_E/tau_s_re : siemens
tau_s_de : second
tau_s_re : second

dg_I/dt = -g_I/tau_s_di + x_I/tau_s_di : siemens
dx_I/dt = -x_I/tau_s_ri : siemens
tau_s_di : second
tau_s_ri : second


dg_E_extnl/dt = -g_E_extnl/tau_s_de_extnl + x_E_extnl/tau_s_de_extnl : siemens
dx_E_extnl/dt = -x_E_extnl/tau_s_re_extnl : siemens
tau_s_de_extnl : second
tau_s_re_extnl : second

I_extnl_crt : amp

'''

synapse_e_AD = '''
w: siemens
'''

synapse_i_AD = '''
w: siemens
'''

