from brian2 import *
from connection import model_neu_syn_AD
from connection import coordination

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pickle
import os

def set_stimulus(lattice,centre,radius): # Find the set of lattice indices for a circular stimulus
    x = lattice[:,0]
    y = lattice[:,1]
    r = ((x-centre[0])**2 + (y-centre[1])**2)**0.5
    stim_index_base = np.argwhere(r<=radius)
    stim_index = [int(i) for i in stim_index_base]
    return stim_index

start_scope()

C = 0.25*nF # nF;  membrane capacitance
g_l = 16.7*nS # nS; leak capacitance
v_l = -70*mV # mV; leak voltage
v_threshold = -50*mV # mV; spike threshold
v_reset = -70*mV # mV;  reset voltage
v_rev_I = -80*mV # mV;  reverse voltage for inhibitory synaptic current
v_rev_E = 0*mV # mV;  reverse voltage for exitatory synaptic current
v_k = -85*mV # mV;  reverse voltage for adaptation potassium current
t_ref = 5*ms # ms;  refractory period

centre = [0,0]
radius = 2
lattice = coordination.makelattice(10,9,[0,0])
stim_index = set_stimulus(lattice,centre,radius)

#P = PoissonGroup(1, 220*2.8*Hz)
P = SpikeGeneratorGroup(1,[0],[1]*ms,period = 15*ms)

G = NeuronGroup(100, model=model_neu_syn_AD.neuron_e_AD,
                     threshold='v>v_threshold', method='euler',
                     reset='''v = v_reset
                              g_k += delta_gk''', refractory='(t-lastspike)<t_ref')

params = {'delta_gk' : 16. * nS,
                    'tau_k' : 60. * ms,
                    'tau_s_di' : 4.4 * ms,
                    'tau_s_de' : 5. * ms,
                    'tau_s_de_inter' : 5. * ms,
                    'tau_s_de_extnl' : 5. * ms,
                    'tau_s_re' : 1. * ms,
                    'tau_s_re_inter' : 1. * ms,
                    'tau_s_re_extnl' : 1. * ms,
                    'tau_s_ri' : 1. * ms,
                    'I_extnl_crt' : 0. * nA,}

for key in params:
    attr = params[key]
    setattr(G, key, attr)

G.v = np.array([-70])*mV#np.random.random(1)*35*mV-85*mV


S_P_G = Synapses(P, G, model=model_neu_syn_AD.synapse_e_AD,
                            on_pre='''x_E_post += w''')

S_P_G.connect(i = list(np.zeros(len(stim_index),int)), j = stim_index)
#S_P_G.connect(i=0,j=0)
S_P_G.w = np.array([1])*120*nS
S_P_G.delay= [1*ms]

volts = StateMonitor(G, 'v', record=[55])
spikes = SpikeMonitor(G, record = True)
net = Network(collect())
net.run(1000*ms)

spike_times = np.round(spikes.t/(1*ms)).astype(int)
spike_index = spikes.i[:]

data = {'i': spike_index, 't': spike_times}

# os.chdir('C:/Users/Evan Xie/Desktop/model_code')
# filename = 'testingOneNeuron.pickle'
# with open(filename, 'wb') as handle:
#     pickle.dump(data, handle)
# handle.close()
# print(filename)

# plt.figure(figsize=(10,6))
# plt.plot(volts.t/ms, volts.v[0])
# plt.xlabel('Time (ms)')
# plt.ylabel('v')
# plt.show()


plt.figure(figsize=(6,6))
plt.plot(lattice[spike_index][:,0],lattice[spike_index][:,1],'.')
plt.show()

