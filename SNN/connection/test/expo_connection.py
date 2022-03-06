#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:58:31 2020

@author: shni2598
"""

import numpy as np
#import brian2.numpy_ as np
#import matplotlib.pyplot as plt
import connection as cn
import warnings
from scipy import sparse
#from brian2.only import *
#import time
#import mydata
#import firing_rate_analysis
#import os
#import datetime
#%%
class get_ijwd:
    
    def __init__(self, *param_dict, **kwargs):
        
        self.Ne = 3969
        self.Ni = 1024
        self.width = 62
        self.decay_p_ee = 8 # decay constant of e to e connection probability as distance increases
        self.decay_p_ei = 10 # decay constant of e to i connection probability as distance increases
        self.decay_p_ie = 20 # decay constant of i to e connection probability as distance increases
        self.decay_p_ii = 20 # decay constant of i to i connection probability as distance increases
        self.mean_SynNumIn_ee = 320     ; # p = 0.08
        self.mean_SynNumIn_ei = 500     ; # p = 0.125
        self.mean_SynNumIn_ie = 200     ; # p = 0.2
        self.mean_SynNumIn_ii = 250     ; # p = 0.25
        self.w_ee_mean = 4
        self.w_ei_mean = 5
        self.w_ie_mean = 20
        self.w_ii_mean = 25
        self.delay = 4
        for dictionary in param_dict:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
        # if either param_dict or kwargs is not empty, reset some parameters which are dependent on other parameters    
        if bool(param_dict) or bool(kwargs): self.change_dependent_para()
    
    def change_dependent_para(self):
        pass
    
    def generate_ijw(self):
        
        self.e_lattice = cn.coordination.makelattice(int(round(np.sqrt(self.Ne))), self.width, [0,0])
        self.i_lattice = cn.coordination.makelattice(int(round(np.sqrt(self.Ni))), self.width, [0,0])
        
        self.p_peak_ee = self.find_peak_probability(self.mean_SynNumIn_ee, self.decay_p_ee, self.e_lattice, self.width, position=[0,0])
        self.p_peak_ei = self.find_peak_probability(self.mean_SynNumIn_ei, self.decay_p_ei, self.e_lattice, self.width, position=[0,0])
        self.p_peak_ie = self.find_peak_probability(self.mean_SynNumIn_ie, self.decay_p_ie, self.i_lattice, self.width, position=[0,0])
        self.p_peak_ii = self.find_peak_probability(self.mean_SynNumIn_ii, self.decay_p_ii, self.i_lattice, self.width, position=[0,0])
        
        self.i_ee, self.j_ee, _ = cn.connect_2lattice.expo_decay(self.e_lattice, self.e_lattice, np.arange(self.Ne), self.width, periodic_boundary=True, interarea_dist=0, \
                                       peak_p=self.p_peak_ee, tau_d=self.decay_p_ee, src_equal_trg = True, self_cnt = False)
        
        self.i_ei, self.j_ei, _ = cn.connect_2lattice.expo_decay(self.e_lattice, self.i_lattice, np.arange(self.Ne), self.width, periodic_boundary=True, interarea_dist=0, \
                                       peak_p=self.p_peak_ei, tau_d=self.decay_p_ei, src_equal_trg = False, self_cnt = False)
        
        self.i_ie, self.j_ie, _ = cn.connect_2lattice.expo_decay(self.i_lattice, self.e_lattice, np.arange(self.Ni), self.width, periodic_boundary=True, interarea_dist=0, \
                                       peak_p=self.p_peak_ie, tau_d=self.decay_p_ie, src_equal_trg = False, self_cnt = False)
        
        self.i_ii, self.j_ii, _ = cn.connect_2lattice.expo_decay(self.i_lattice, self.i_lattice, np.arange(self.Ni), self.width, periodic_boundary=True, interarea_dist=0, \
                                       peak_p=self.p_peak_ii, tau_d=self.decay_p_ii, src_equal_trg = True, self_cnt = False)

        self.num_in_ee = self.find_indegree(self.i_ee, self.j_ee, self.Ne, self.Ne)
        self.num_in_ei = self.find_indegree(self.i_ei, self.j_ei, self.Ne, self.Ni)
        self.num_in_ie = self.find_indegree(self.i_ie, self.j_ie, self.Ni, self.Ne)
        self.num_in_ii = self.find_indegree(self.i_ii, self.j_ii, self.Ni, self.Ni)
        
        self.J_ee = self.find_J(self.num_in_ee, self.w_ee_mean)
        self.J_ei = self.find_J(self.num_in_ei, self.w_ei_mean)
        self.J_ie = self.find_J(self.num_in_ie, self.w_ie_mean)
        self.J_ii = self.find_J(self.num_in_ii, self.w_ii_mean)
        
        self.w_ee = self.find_synWeight(self.j_ee, self.J_ee, self.Ne, self.num_in_ee)
        self.w_ei = self.find_synWeight(self.j_ei, self.J_ei, self.Ni, self.num_in_ei)
        self.w_ie = self.find_synWeight(self.j_ie, self.J_ie, self.Ne, self.num_in_ie)
        self.w_ii = self.find_synWeight(self.j_ii, self.J_ii, self.Ni, self.num_in_ii)
    
    def generate_d_rand(self):
        
        self.d_ee = np.random.uniform(0, self.delay, len(self.i_ee))
        self.d_ie = np.random.uniform(0, self.delay, len(self.i_ie))
        self.d_ei = np.random.uniform(0, self.delay, len(self.i_ei))
        self.d_ii = np.random.uniform(0, self.delay, len(self.i_ii))

    def find_peak_probability(self, mean_SynNum, tau_d, lattice, width, position=[0,0]):
    
        dist = cn.coordination.lattice_dist(lattice, width, position)
        p_peak = mean_SynNum/(np.sum(np.exp(-dist/tau_d)))
        if p_peak > 1:
            warnings.warn(" Error: peak connection probability exceeds '1', adjust 'decay_p' or 'mean_SynSum' to reduce it below '1'" )
            
        return p_peak
    
    def find_indegree(self, i, j, N_src, N_trg):
        
        A = sparse.csc_matrix((np.ones(i.shape, dtype=int), (i, j)),shape=(N_src,N_trg))
        num_in = A.sum(0)    
        return num_in.A[0]

    def find_J(self, num_in, J_mean):
        J = J_mean*np.sum(num_in)/(np.sum(np.sqrt(num_in)))
        return J 

    def find_synWeight(self, j_cnt, J_weight, N_trg, num_indegree):
    
        w_cnt = np.zeros(j_cnt.shape)
        for j in range(N_trg):
            J_mean = J_weight/np.sqrt(num_indegree[j])
            J_std = J_mean/4
            w_tmp = np.random.randn(num_indegree[j])*J_std + J_mean
            negative_w = np.any(w_tmp <= 0)
            while negative_w:
                #w_tmp = np.random.randn(num_indegree[j])*J_std + J_mean
                w_tmp[w_tmp <= 0] = np.random.randn(np.sum(w_tmp <= 0))*J_std + J_mean
                negative_w = np.any(w_tmp <= 0)
                
            w_cnt[j_cnt == j] = w_tmp
                
        return w_cnt
#%%
"""
def find_peak_probability(mean_SynNum, tau_d, lattice, width, position=[0,0]):
    
    dist = cn.coordination.lattice_dist(lattice, width, position)
    p_peak = mean_SynNum/(np.sum(np.exp(-dist/tau_d)))
    if p_peak > 1:
        warnings.warn(" Error: peak connection probability exceeds '1', adjust 'tau_d' or 'mean_SynSum' to reduce it below '1'" )
    
    return p_peak

#%%
e_lattice = cn.coordination.makelattice(63, 62, [0,0])
i_lattice = cn.coordination.makelattice(32, 62, [0,0])
#%%
tau_d_ee = 8             ;tau_d_ei = 10             ;tau_d_ie = 20             ;tau_d_ii = 20             ;
mean_SynNum_ee = 320     ;mean_SynNum_ei = 500      ;mean_SynNum_ie = 200      ;mean_SynNum_ii = 250      ;

width = 62
#%%

p_peak_ee = find_peak_probability(mean_SynNum_ee, tau_d_ee, e_lattice, width, position=[0,0])
p_peak_ei = find_peak_probability(mean_SynNum_ei, tau_d_ei, e_lattice, width, position=[0,0])
p_peak_ie = find_peak_probability(mean_SynNum_ie, tau_d_ie, i_lattice, width, position=[0,0])
p_peak_ii = find_peak_probability(mean_SynNum_ii, tau_d_ii, i_lattice, width, position=[0,0])
#%%
Ne = 3969;
Ni = 1024;
i_ee, j_ee, _ = cn.connect_2lattice.expo_decay(e_lattice, e_lattice, np.arange(Ne), 62, periodic_boundary=True, interarea_dist=0, \
                               peak_p=p_peak_ee, tau_d=tau_d_ee, src_equal_trg = True, self_cnt = False)

i_ei, j_ei, _ = cn.connect_2lattice.expo_decay(e_lattice, i_lattice, np.arange(Ne), 62, periodic_boundary=True, interarea_dist=0, \
                               peak_p=p_peak_ei, tau_d=tau_d_ei, src_equal_trg = False, self_cnt = False)

i_ie, j_ie, _ = cn.connect_2lattice.expo_decay(i_lattice, e_lattice, np.arange(Ni), 62, periodic_boundary=True, interarea_dist=0, \
                               peak_p=p_peak_ie, tau_d=tau_d_ie, src_equal_trg = False, self_cnt = False)

i_ii, j_ii, _ = cn.connect_2lattice.expo_decay(i_lattice, i_lattice, np.arange(Ni), 62, periodic_boundary=True, interarea_dist=0, \
                               peak_p=p_peak_ii, tau_d=tau_d_ii, src_equal_trg = True, self_cnt = False)


#%%
def find_indegree(i, j, N_src, N_trg):
    A = sparse.csc_matrix((np.ones(i.shape, dtype=int), (i, j)),shape=(N_src,N_trg))
    num_in = A.sum(0)
    return num_in.A[0]
#%%
num_in_ee = find_indegree(i_ee, j_ee, Ne, Ne)
num_in_ei = find_indegree(i_ei, j_ei, Ne, Ni)
num_in_ie = find_indegree(i_ie, j_ie, Ni, Ne)
num_in_ii = find_indegree(i_ii, j_ii, Ni, Ni)


#%%
def find_J(num_in, J_mean):
    J = J_mean*np.sum(num_in)/(np.sum(np.sqrt(num_in)))
    return J 
#%%
w_ee_mean = 4
w_ei_mean = 5
w_ie_mean = 20
w_ii_mean = 25

J_ee = find_J(num_in_ee, w_ee_mean)
J_ei = find_J(num_in_ei, w_ei_mean)
J_ie = find_J(num_in_ie, w_ie_mean)
J_ii = find_J(num_in_ii, w_ii_mean)

#%%
def find_synWeight(j_cnt, J_weight, N_trg, num_indegree):
    
    w_cnt = np.zeros(j_cnt.shape)
    for j in range(N_trg):
        J_mean = J_weight/np.sqrt(num_indegree[j])
        J_std = J_mean/4
        w_tmp = np.random.randn(num_indegree[j])*J_std + J_mean
        negative_w = np.any(w_tmp <= 0)
        while negative_w:
            w_tmp = np.random.randn(num_indegree[j])*J_std + J_mean
            w_tmp[w_tmp <= 0] = np.random.randn(np.sum(w_tmp <= 0))*J_std + J_mean
            negative_w = np.any(w_tmp <= 0)
            
        w_cnt[j_cnt == j] = w_tmp
    
    return w_cnt
#%%
w_ee = find_synWeight(j_ee, J_ee, Ne, num_in_ee)
w_ei = find_synWeight(j_ei, J_ei, Ni, num_in_ei)
w_ie = find_synWeight(j_ie, J_ie, Ne, num_in_ie)
w_ii = find_synWeight(j_ii, J_ii, Ni, num_in_ii)



#%%
plt.figure()
plt.hist(num_in_ie)
#%%
#%%%%%%%%%%%%%%%%%%%%%%
e_lattice = cn.coordination.makelattice(63, 62, [0,0])
i_lattice = cn.coordination.makelattice(32, 62, [0,0])
"""
#%%

"""
ijwd = get_ijwd()
ijwd.Ne = 77*77; ijwd.Ni = 39*39
ijwd.width = 76
ijwd.generate_ijw()
ijwd.generate_d_rand()
#%%

chg_adapt_loca = [0, 0]
chg_adapt_range = 6
chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(ijwd.e_lattice, chg_adapt_loca, chg_adapt_range, ijwd.width)



#%%
start_scope()

neuronmodel_e = cn.model_neu_syn_AD.neuron_e_AD
neuronmodel_i = cn.model_neu_syn_AD.neuron_i_AD

synapse_e = cn.model_neu_syn_AD.synapse_e_AD
synapse_i = cn.model_neu_syn_AD.synapse_i_AD

group_e_1 =NeuronGroup(ijwd.Ne, model=neuronmodel_e,
                     threshold='v>v_threshold', method='euler',
                     reset='''v = v_reset
                              g_k += delta_gk''', refractory='(t-lastspike)<t_ref')

group_i_1 =NeuronGroup(ijwd.Ni, model=neuronmodel_i,
                     threshold='v>v_threshold', method='euler',
                     reset='v = v_reset', refractory='(t-lastspike)<t_ref')

syn_ee_1 = Synapses(group_e_1, group_e_1, model=synapse_e, 
                  on_pre='x_E_post += w')
syn_ei_1 = Synapses(group_e_1, group_i_1, model=synapse_e, 
                  on_pre='x_E_post += w')
syn_ie_1 = Synapses(group_i_1, group_e_1, model=synapse_i, 
                  on_pre='x_I_post += w')
syn_ii_1 = Synapses(group_i_1, group_i_1, model=synapse_i, 
                  on_pre='x_I_post += w')

'''external input'''
#stim_rate = psti.input_spkrate(maxrate = [800,800], sig=[6,6], position=[[0, 0],[31.5,31.5]])*Hz
bkg_rate2e = 850*Hz
bkg_rate2i = 1000*Hz
extnl_e = PoissonGroup(ijwd.Ne, bkg_rate2e)
extnl_i = PoissonGroup(ijwd.Ni, bkg_rate2i)

#tau_x_re = 1*ms
syn_extnl_e = Synapses(extnl_e, group_e_1, model=synapse_e, on_pre='x_E_extnl_post += w')
syn_extnl_i = Synapses(extnl_i, group_i_1, model=synapse_e, on_pre='x_E_extnl_post += w')

syn_extnl_e.connect('i==j')
syn_extnl_i.connect('i==j')

w_extnl_ = 1.5 # nS
syn_extnl_e.w = w_extnl_*5.8*nS#*tau_s_de_*nS
syn_extnl_i.w = w_extnl_*5.8*nS#*tau_s_de_*nS


syn_ee_1.connect(i=ijwd.i_ee, j=ijwd.j_ee)
syn_ei_1.connect(i=ijwd.i_ei, j=ijwd.j_ei)
syn_ie_1.connect(i=ijwd.i_ie, j=ijwd.j_ie)
syn_ii_1.connect(i=ijwd.i_ii, j=ijwd.j_ii)


#tau_s_di_try = tau_s_di_
syn_ee_1.w = ijwd.w_ee*nsiemens * 5.8 #tau_s_de_
syn_ei_1.w = ijwd.w_ei*nsiemens * 5.8 #tau_s_de_ #5*nS
syn_ii_1.w = ijwd.w_ii*nsiemens * 6. #tau_s_di_#25*nS
syn_ie_1.w = ijwd.w_ie*nsiemens * 6. #tau_s_di_#
#w_ext = 2*nS
#syn_pois_e.w = w_ext
#syn_pois_i.w = w_ext

def set_delay(syn, delay_up):
    #n = len(syn)
    syn.delay = delay_up*ms
    #syn.down.delay = (delay_up + 1)*ms
    
    return syn 

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
syn_ee_1 = set_delay(syn_ee_1, ijwd.d_ee)
syn_ie_1 = set_delay(syn_ie_1, ijwd.d_ie)
syn_ei_1 = set_delay(syn_ei_1, ijwd.d_ei)
syn_ii_1 = set_delay(syn_ii_1, ijwd.d_ii)

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

group_i_1.tau_s_de = tau_s_de_*ms
group_i_1.tau_s_di = tau_s_di_*ms
group_i_1.tau_s_re = group_i_1.tau_s_ri = 1*ms

group_i_1.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
group_i_1.tau_s_re_inter = 1*ms
group_i_1.tau_s_de_extnl = tau_s_de_*ms #5.0*ms
group_i_1.tau_s_re_extnl = 1*ms

#syn_pois_e.effect = 0
#syn_pois_i.effect = 0
#group_e_v2.v = np.random.random(Ne)*35*mV-85*mV
#group_i_v2.v = np.random.random(Ni)*35*mV-85*mV
#seed(1000)
group_e_1.v = np.random.random(ijwd.Ne)*35*mV-85*mV
group_i_1.v = np.random.random(ijwd.Ni)*35*mV-85*mV
group_e_1.delta_gk = delta_gk_*nS
group_e_1.tau_k = 80*ms
#group_e_v2.v = np.random.random(Ne)*10*mV-60*mV
#group_i_v2.v = np.random.random(Ni)*10*mV-60*mV
group_e_1.I_extnl_crt = 0*nA #0.51*nA
group_i_1.I_extnl_crt = 0*nA #0.60*nA

#%%
spk_e = SpikeMonitor(group_e_1, record = True)
spk_i = SpikeMonitor(group_i_1, record = True)
#lfp_moni = StateMonitor(group_LFP_record, ('lfp1','lfp2','lfp3'), record = True)
#%%
net = Network(collect())
net.store('state1')
#%%
scale_ee = 1.; 
scale_ei = 1.03
scale_ie = 1.
scale_ii = 0.9
syn_ee_1.w = ijwd.w_ee*nsiemens * 5.8 * scale_ee#tau_s_de_
syn_ei_1.w = ijwd.w_ei*nsiemens * 5.8 * scale_ei #tau_s_de_ #5*nS
syn_ii_1.w = ijwd.w_ii*nsiemens * 6. * scale_ie#tau_s_di_#25*nS
syn_ie_1.w = ijwd.w_ie*nsiemens * 6. * scale_ii#tau_s_di_#
#%%
#change_ie(4.4)
#syn_ie_1.w = w_ie*usiemens

print('ie_w: %fnsiemens' %(syn_ie_1.w[0]/nsiemens))
#Ne = 63*63; Ni = 1000;
C = 0.25*nF # capacitance
g_l = 16.7*nS # leak capacitance
v_l = -70*mV # leak voltage
v_threshold = -50*mV
v_reset = -60*mV
v_rev_I = -80*mV
v_rev_E = 0*mV
v_k = -85*mV
#tau_k = 80*ms# 80*ms
#delta_gk = 10*nS #10*nS
t_ref = 4*ms # refractory period
new_delta_gk_ = 0
#tau_s_de = 5*ms
#tau_s_di = 3*ms
#tau_s_re = 1*ms
#tau_s_ri = 1*ms
#tau_x_re = 1*ms
tic = time.perf_counter()
#seed(10)
simu_time1 = 1000*ms#2000*ms
simu_time2 = 2000*ms
#simu_time2 = 2000*ms#8000*ms
#simu_time3 = 1000*ms
#simu_time_tot = 30000*ms
#group_input.active = False
extnl_e.rates = bkg_rate2e
net.run(simu_time1, profile=False) #,namespace={'tau_k': 80*ms}
#extnl_e.rates = bkg_rate2e + stim_rate
group_e_1.delta_gk[chg_adapt_neuron] = new_delta_gk_*nS
net.run(simu_time2, profile=False) #,namespace={'tau_k': 80*ms}
#extnl_e.rates = bkg_rate2e
#net.run(simu_time3, profile=False) #,namespace={'tau_k': 80*ms}
#group_input.active = True
#group_e_1.delta_gk[chg_adapt_neuron] = 2*nS; group_e_1.tau_k[chg_adapt_neuron] = 40*ms
#net.run(simu_time2, report = 'text', profile=False) #,namespace={'tau_k': 80*ms}
print('total time elapsed:',time.perf_counter() - tic)
#%%
spk_tstep_e = np.round(spk_e.t/(0.1*ms)).astype(int)
spk_tstep_i = np.round(spk_i.t/(0.1*ms)).astype(int)
now = datetime.datetime.now()
loop_num = 0
#param_1 = {'delta_gk':10,'tau_s_di':3,'tau_s_de':5,'ie_ratio':3.1725,'sti_onset':'10s'}
'''
param_new = {'delta_gk':delta_gk_,
         'new_delta_gk':new_delta_gk_,
         #'new_tau_k':40,
         'tau_s_di':tau_s_di_,
         'tau_s_de':tau_s_de_, 'ie_ratio':ie_ratio_,
         'mean_J_ee': ijwd.mean_J_ee,
         #'chg_adapt_range':6, 
         'p_ee':p_ee,
         'simutime':simu_time_tot/ms,
         'chg_adapt_time': simu_time1/ms,
         'chg_adapt_range': chg_adapt_range,
         'chg_adapt_loca': chg_adapt_loca,}
param = {**param, **param_new}
'''
param = {}
#param_12 = {'scale_e_12':scale_e_12[i], 'scale_e_21':scale_e_21[j]}

#data = {'param':param_12, 'a1':{'param':param_1,'e':{'i':spk_1.i[:],'t':spk_1_tstep}},
#                            'a2':{'param':param_2,'e':{'i':spk_2.i[:],'t':spk_2_tstep}}}
data = {'datetime':now.strftime("%Y-%m-%d %H:%M:%S"), 'dt':0.1, 'loop_num':loop_num, 'data_dir': os.getcwd(),
        'a1':{'param':param,
              #'LFP':{'lfp1':lfp_moni.lfp1/nA, 'lfp2':lfp_moni.lfp2/nA, 'lfp3':lfp_moni.lfp3/nA},
              'ge':{'i':spk_e.i[:],'t':spk_tstep_e},
              'gi':{'i':spk_i.i[:],'t':spk_tstep_i}}}


data1 = mydata.mydata(data)
#%%
data1.a1.ge.get_spike_rate(start_time=500, end_time=3000,\
                           sample_interval = 1, n_neuron = ijwd.Ne, window = 10, dt = 0.1, reshape_indiv_rate = True)
data1.a1.gi.get_spike_rate(start_time=500, end_time=3000,\
                           sample_interval = 1, n_neuron = ijwd.Ni, window = 10, dt = 0.1, reshape_indiv_rate = True)

firing_rate_analysis.show_pattern(data1.a1.ge.spk_rate.spk_rate, data1.a1.gi.spk_rate.spk_rate, frames = 2500, start_time = 500, anititle='animation')

#%%
plt.figure()
plt.plot(data1.a1.ge.t/10,data1.a1.ge.i,'|')
#%%
plt.figure()
plt.imshow(data1.a1.ge.spk_rate.spk_rate[:,:,100])
#%%
plt.figure()
plt.scatter( ijwd.e_lattice[:,0], ijwd.e_lattice[:,1])
plt.figure()
plt.scatter( ijwd.i_lattice[:,0], ijwd.i_lattice[:,1])
#%%
pre_neuron = 5500
pre = ijwd.e_lattice[pre_neuron]; post = ijwd.e_lattice[ijwd.j_ee[ijwd.i_ee==pre_neuron]]
plt.figure()
plt.plot(post[:,0],post[:,1], '.')
#%%
pre_neuron = 1000
pre = ijwd.i_lattice[pre_neuron]; post = ijwd.e_lattice[ijwd.j_ie[ijwd.i_ie==pre_neuron]]
plt.figure()
plt.plot(post[:,0],post[:,1], '.')
#%%
pre_neuron = 1000
pre = ijwd.i_lattice[pre_neuron]; post = ijwd.i_lattice[ijwd.j_ii[ijwd.i_ii==pre_neuron]]
plt.figure()
plt.plot(post[:,0],post[:,1], '.')
#%%
net.restore('state1')

#%%
ijwd_63 = get_ijwd()
ijwd_63.Ne = 63*63; ijwd_63.Ni = 32*32
ijwd_63.width = 62
ijwd_63.generate_ijw()
ijwd_63.generate_d_rand()
#%%
def plot_outgoing(i, j, neuron_src, lattice_src, lattice_trg):
    #pre_neuron = 5500
    pre = lattice_src[neuron_src]; post = lattice_trg[j[i==neuron_src]]
    plt.figure()
    plt.plot(post[:,0],post[:,1], '.')
    plt.plot(pre[0], pre[1], 'or')
#%%
plot_outgoing(ijwd_63.i_ie, ijwd_63.j_ie, 0, ijwd_63.i_lattice, ijwd_63.e_lattice)   
plot_outgoing(ijwd.i_ie, ijwd.j_ie, 0, ijwd.i_lattice, ijwd.e_lattice)   
   
#%%
plot_outgoing(ijwd_63.i_ii, ijwd_63.j_ii, 0, ijwd_63.i_lattice, ijwd_63.i_lattice)   
plot_outgoing(ijwd.i_ii, ijwd.j_ii, 0, ijwd.i_lattice, ijwd.i_lattice)   
"""
    


