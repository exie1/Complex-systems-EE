# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 20:52:55 2020

@author: nishe
"""

neuron_e_YG = '''
dv/dt = (1/C)*(-g_l*(v - v_l) + (-g_k)*(v - v_k) + I_inh + I_exi + I_extnl_spk + I_extnl_crt) : volt (unless refractory)

I_inh = (-g_I)*(v - v_rev_I) : amp
I_exi = (-g_E-g_E_inter)*(v - v_rev_E) : amp
I_extnl_spk = (-g_E_extnl)*(v - v_rev_E) : amp

dg_k/dt = -g_k/tau_k : siemens
delta_gk : siemens
tau_k : second

g_I : siemens
g_E : siemens
g_E_inter : siemens
I_extnl_crt : amp

g_E_extnl = w_extnl*s_extnl: siemens
w_extnl: siemens
ds_extnl/dt = -s_extnl/tau_s_de_extnl + x_extnl : 1
dx_extnl/dt = -x_extnl/tau_s_re_extnl : Hz
tau_s_de_extnl : second
tau_s_re_extnl : second
'''

neuron_i_YG = '''
dv/dt = (1/C)*(-g_l*(v - v_l) + I_inh + I_exi + I_extnl_spk + I_extnl_crt) : volt (unless refractory)
I_inh = (-g_I)*(v - v_rev_I) : amp
I_exi = (-g_E-g_E_inter)*(v - v_rev_E) : amp
I_extnl_spk = (-g_E_extnl)*(v - v_rev_E) : amp

g_I : siemens
g_E : siemens
g_E_inter : siemens
I_extnl_crt : amp

g_E_extnl = w_extnl*s_extnl: siemens
w_extnl: siemens
ds_extnl/dt = -s_extnl/tau_s_de_extnl + x_extnl : 1
dx_extnl/dt = -x_extnl/tau_s_re_extnl : Hz
tau_s_de_extnl : second
tau_s_re_extnl: second
'''

synapse_e_YG = '''
w: siemens
g_E_post = w*s : siemens (summed)
ds/dt = -s/tau_s_de + rect_puls*(1 - s) : 1 (clock-driven)
rect_puls = (1/tau_s_re)*effect : Hz
effect : integer
tau_s_de : second
tau_s_re : second
'''

synapse_i_YG = '''
w: siemens
g_I_post = w*s : siemens (summed)
ds/dt = -s/tau_s_di + rect_puls*(1 - s) : 1 (clock-driven)
rect_puls = (1/tau_s_ri)*effect : Hz
effect : integer
tau_s_di : second
tau_s_ri : second
'''
#%%
# inter-area synapse
synapse_e_inter_YG = '''
w: siemens
g_E_inter_post = w*s : siemens (summed)
ds/dt = -s/tau_s_de + rect_puls*(1 - s) : 1 (clock-driven)
rect_puls = (1/tau_s_re)*effect : Hz
effect : integer
tau_s_de : second
tau_s_re : second
'''


