# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 21:13:03 2020

@author: nishe
"""

neuron_e_AD = '''
dv/dt = (1/C)*(-g_l*(v - v_l) + (-g_k)*(v - v_k) + I_inh + I_exi + I_extnl_spk + I_extnl_crt) : volt (unless refractory)

I_inh = (-g_I)*(v - v_rev_I) : amp
I_exi = (-g_E - g_E_inter)*(v - v_rev_E): amp
I_extnl_spk = (-g_E_extnl)*(v - v_rev_E) : amp

dg_k/dt = -g_k/tau_k :siemens
delta_bkg : siemens
delta_stim : siemens
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

synapse_e_AD_SP = '''
w: siemens
du/dt = (u_base - u)/tau_f : 1 (event-driven)
dL/dt = (1 - L)/tau_d : 1 (event-driven)
'''

synapse_i_AD_SP = '''
w: siemens
du/dt = (u_base - u)/tau_f : 1 (event-driven)
dL/dt = (1 - L)/tau_d : 1 (event-driven)
'''
#%%
'''enable delta_gk to be time-varying'''

neuron_e_AD_adapt_tv = '''
dv/dt = (1/C)*(-g_l*(v - v_l) + (-g_k)*(v - v_k) + I_inh + I_exi + I_extnl_spk + I_extnl_crt) : volt (unless refractory)

I_inh = (-g_I)*(v - v_rev_I) : amp
I_exi = (-g_E - g_E_inter)*(v - v_rev_E): amp
I_extnl_spk = (-g_E_extnl)*(v - v_rev_E) : amp

dg_k/dt = -g_k/tau_k :siemens
delta_gk = delta_gk_base + delta_gk_modu*dgk_modu_on(t) : siemens
delta_gk_base : siemens
delta_gk_modu : siemens

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

#%%
""""test, C g_l, t_ref"""
neuron_e_AD_t = '''
dv/dt = (1/C)*(-g_l*(v - v_l) + (-g_k)*(v - v_k) + I_inh + I_exi + I_extnl_spk + I_extnl_crt) : volt (unless refractory)
C: farad
g_l: siemens
t_ref : second

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

neuron_i_AD_t = '''
dv/dt = (1/C)*(-g_l*(v - v_l) + I_inh + I_exi + I_extnl_spk + I_extnl_crt) : volt (unless refractory)
C: farad
g_l: siemens
t_ref : second

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