# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 15:51:13 2020

@author: nishe
"""


#%%
import connection as cn
import numpy as np
#%%

class get_inter_ijwd:
    
    def __init__(self, *param_dict, **kwargs):
        
        self.Ne = 64*64 # excitatory neurons in both areas
        self.Ni = 1024 # inhibitory neurons in both areas
        self.width = 63 #width of lattice on both areas
        self.delay_12 = [8,10] # spike transmission delay between two areas (ms)

        self.periodic_boundary = True; # if use the periodic_boundary for inter-area synapse connection
        self.interarea_dist = 0; # inter-area distance
        # peak inter-area connection probability for 1e-2e, 1e-2i, 2e-1e and 2e-1i  
        self.peak_p_1e_2e = 0.85; self.peak_p_1e_2i = 0.85;
        self.peak_p_2e_1e = 0.4; self.peak_p_2e_1i = 0.4; 
        # decay constant of inter-area connection probability
        self.tau_d_1e_2e = 5; self.tau_d_1e_2i = 5
        self.tau_d_2e_1e = 8; self.tau_d_2e_1i = 8
    
        self.n_inter_area_1 = int(0.5*self.Ne) # number of excitatory neurons form inter-areal projections in area 1
        self.n_inter_area_2 = int(0.5*self.Ne) # number of excitatory neurons form inter-areal projections in area 2
       
        self.inter_e_neuron_1 = np.random.choice(self.Ne, self.n_inter_area_1, replace = False) # excitatory neurons forming inter-areal projections in area 1
        self.inter_e_neuron_1 = np.sort(self.inter_e_neuron_1) # excitatory neurons forming inter-areal projections in area 2
        
        self.inter_e_neuron_2 = np.random.choice(self.Ne, self.n_inter_area_2, replace = False)
        self.inter_e_neuron_2 = np.sort(self.inter_e_neuron_2)
                
        for dictionary in param_dict:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
        
        if bool(param_dict) or bool(kwargs): self.change_dependent_para()

    
    def change_dependent_para(self):
        
        self.n_inter_area_1 = int(0.5*self.Ne) # number of excitatory neurons forming inter-areal projections in area 1
        self.n_inter_area_2 = int(0.5*self.Ne) # number of excitatory neurons forming inter-areal projections in area 2
       
        self.inter_e_neuron_1 = np.random.choice(self.Ne, self.n_inter_area_1, replace = False)
        self.inter_e_neuron_1 = np.sort(self.inter_e_neuron_1)
        
        self.inter_e_neuron_2 = np.random.choice(self.Ne, self.n_inter_area_2, replace = False)
        self.inter_e_neuron_2 = np.sort(self.inter_e_neuron_2)

    
    def generate_ij(self, ijwd1, ijwd2):
        
        self.i_1e_2e, self.j_1e_2e, self.dist_1e_2e = cn.connect_2lattice.expo_decay(ijwd1.lattice_ext, ijwd2.lattice_ext, self.inter_e_neuron_1, \
                                                              self.width, self.periodic_boundary, self.interarea_dist, \
                                                              self.peak_p_1e_2e, self.tau_d_1e_2e, src_equal_trg = False, self_cnt = False)
        
        self.i_1e_2i, self.j_1e_2i, self.dist_1e_2i = cn.connect_2lattice.expo_decay(ijwd1.lattice_ext, ijwd2.lattice_inh, self.inter_e_neuron_1, \
                                                              self.width, self.periodic_boundary, self.interarea_dist, \
                                                              self.peak_p_1e_2i, self.tau_d_1e_2i, src_equal_trg = False, self_cnt = False)
        
        self.i_2e_1e, self.j_2e_1e, self.dist_2e_1e = cn.connect_2lattice.expo_decay(ijwd2.lattice_ext, ijwd1.lattice_ext, self.inter_e_neuron_2, \
                                                              self.width, self.periodic_boundary, self.interarea_dist, \
                                                              self.peak_p_2e_1e, self.tau_d_2e_1e, src_equal_trg = False, self_cnt = False)
        
        self.i_2e_1i, self.j_2e_1i, self.dist_2e_1i = cn.connect_2lattice.expo_decay(ijwd2.lattice_ext, ijwd1.lattice_inh, self.inter_e_neuron_2, \
                                                              self.width, self.periodic_boundary, self.interarea_dist, \
                                                              self.peak_p_2e_1i, self.tau_d_2e_1i, src_equal_trg = False, self_cnt = False)
    
    def generate_d_rand(self):
        
        self.d_1e_2e = np.random.uniform(self.delay_12[0], self.delay_12[1], len(self.i_1e_2e))
        self.d_1e_2i = np.random.uniform(self.delay_12[0], self.delay_12[1], len(self.i_1e_2i))
        self.d_2e_1e = np.random.uniform(self.delay_12[0], self.delay_12[1], len(self.i_2e_1e))
        self.d_2e_1i = np.random.uniform(self.delay_12[0], self.delay_12[1], len(self.i_2e_1i))
        
#%%
    
    
    
    
    
    
