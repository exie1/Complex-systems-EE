#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:35:05 2019

@author: shni2598
"""

from connection import g_pool_generator
from connection import hybrid_degree
from connection import generate_connection
from connection import inverse_pool
from connection import connect_2lattice
from connection import quasi_lattice
from connection import coordination
from connection import shuffle_weight_common_neighbour
from connection import set_workingmemory
import time
#import brian2.numpy_ as np
import numpy as np
from scipy.sparse import csr_matrix

#%%
class get_ijwd:
    
    #cls_var = ['a'];
    
    def __init__(self, *param_dict, **kwargs):
        
        self.Ni = 1000; self.Ne = 63*63 # number of i and e neurons

        self.p_ee = 0.16; self.p_ie = 0.2 # connection probability
        self.p_ei = 0.2; self.p_ii = 0.4
        
        self.degree_mean_ee = self.Ne * self.p_ee # mean of in- and out-degree
        self.cv = 0.2
        self.degree_std = self.cv * self.degree_mean_ee # standard deviation of lognormal distribution of in- and out-degree
        self.r = 0.13 # correlation of in- and out-degree
        self.hybrid = 0.4 # percent of lognormal part of the in- and out-degree 
        
        self.iter_num = 5 # number of iterations
        self.cn_scale_wire = 2 # common-neighbour factor for gennerating connectivity (synapses)
        self.cn_scale_weight = 2 # common-neighbour factor for generating synapses weight
        self.record_cc=False
        
        self.width = 62
        #self.width = int(self.Ne**0.5) - 1 # width of exciatory neurons grid
        #tau_d = 8
        self.tau_p_ee = 8 # decay constant of e to e connection probability as distance increases
        self.tau_p_ei = 10 # decay constant of e to i connection probability as distance increases
        self.tau_p_ie = 20 # decay constant of i to e connection probability as distance increases
        self.tau_p_ii = 20 # decay constant of i to i connection probability as distance increases
        self.delay = 4
        self.ie_ratio = 3.375
        self.mean_J_ee = 4*10**-3 # usiemens
        self.sigma_J_ee = 1.9*10**-3 # usiemens
        self.w_ee_dist = 'normal' # 'lognormal'
        
        self.w_ei, self.w_ii = 5*10**(-3), 25*10**(-3) # synapse weight of E-I and I-I connections
        
        self.enable_workingmemory = False
        #self.c = self.test(self.Ni, self.Ne)
#        defa = vars(self)
#        self.default_param = defa       
        for dictionary in param_dict:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
        # if either param_dict or kwargs is not empty, reset some parameters which are dependent on other parameters    
        if bool(param_dict) or bool(kwargs): self.change_dependent_para()
    
    def change_dependent_para(self):
        self.degree_mean_ee = self.Ne * self.p_ee
        self.degree_std = self.cv * self.degree_mean_ee
        #self.width = int(self.Ne**0.5) - 1 # width of exciatory neurons grid
#        self.tau_p_ee = self.width/(int(self.Ne**0.5) - 1)*8 # decay constant of e to e connection probability as distance increases
#        self.tau_p_ie = self.width/(int(self.Ne**0.5) - 1)*20 # decay constant of i to e connection probability as distance increases
#        self.tau_p_ei = self.width/(int(self.Ne**0.5) - 1)*10 # decay constant of e to i connection probability as distance increases
#        self.tau_p_ii = self.width/(int(self.Ne**0.5) - 1)*20 # decay constant of i to i connection probability as distance increases
       
    def add_workingmemory(self, num_mem, neuron_per_mem, position_mem, scale_mem_w):
        #self.add_workingmemory = False
        self.enable_workingmemory = True
        self.num_mem = num_mem; self.neuron_per_mem = neuron_per_mem
        self.position_mem = position_mem; self.scale_mem_w = scale_mem_w   
                #%% add workingmemory
        #set_workingmemory.set_memory_neuronneuron_weight(self.num_mem, self.neuron_per_mem, self.position_mem, self.scale_mem_w)
       
    def generate_ijw(self):
#        self.degree_mean_ee = self.Ne * self.p_ee # mean of in- and out-degree
#        self.degree_std = self.cv * self.degree_mean_ee # standard deviation of lognormal distribution of in- and out-degree
        self.i_ee, self.j_ee, self.w_ee, self.lattice_ext, self.dist_ee = self.e_e(self.Ne, self.degree_mean_ee, self.degree_std, self.r, self.hybrid, self.width, \
                                                   self.tau_p_ee, self.cn_scale_wire, self.iter_num, self.record_cc, \
                                                   self.mean_J_ee, self.sigma_J_ee, self.cn_scale_weight)
    #def generate_ie(self):     
        if self.enable_workingmemory == True:
            print('setting working memory')
            self.w_ee, self.memory_neuron, self.memory_neuron_syn = set_workingmemory.set_memory_syn_weight_preprocess(self.i_ee, self.j_ee, self.w_ee, self.lattice_ext, self.num_mem, self.neuron_per_mem, self.position_mem, self.scale_mem_w)

        self.i_ie, self.j_ie, self.w_ie, self.lattice_inh, self.dist_ie = self.i_e(self.Ne, self.Ni, self.p_ie, self.i_ee, self.j_ee, self.width, \
                                                    self.w_ee, self.ie_ratio, self.lattice_ext, self.tau_p_ie)
        
        self.i_ei, self.j_ei, self.dist_ei, self.i_ii, self.j_ii, self.dist_ii = self.e_i_and_i_e(self.Ne, self.Ni, self.p_ei, self.p_ii, self.lattice_ext, \
                                                                                                  self.lattice_inh, self.tau_p_ei, self.tau_p_ii, self.width)
        
        
        self.w_ei, self.w_ii = self.w_ei, self.w_ii
    
    def generate_d_rand(self):
        
        self.d_ee = np.random.uniform(0, self.delay, len(self.i_ee))
        self.d_ie = np.random.uniform(0, self.delay, len(self.i_ie))
        self.d_ei = np.random.uniform(0, self.delay, len(self.i_ei))
        self.d_ii = np.random.uniform(0, self.delay, len(self.i_ii))
        
    def generate_d_dist(self):
        
        self.d_ee = self.dist_ee/(2**0.5*self.width/2)*self.delay
        self.d_ie = self.dist_ie/(2**0.5*self.width/2)*self.delay
        self.d_ei = self.dist_ei/(2**0.5*self.width/2)*self.delay
        self.d_ii = self.dist_ii/(2**0.5*self.width/2)*self.delay
    
    def generate_d_const(self, delay_t):
        
        self.d_ee = delay_t
        self.d_ie = delay_t
        self.d_ei = delay_t
        self.d_ii = delay_t
        
    def e_e(self, Ne, degree_mean_ee, degree_std, r, hybrid, width, tau_p_ee, cn_scale_wire, iter_num, record_cc, mean_J_ee, sigma_J_ee, cn_scale_weight):
        
        in_degree0, out_degree0 = np.ones(1)*(Ne + 1), np.ones(1)*(Ne + 1)
        coef = 0
        while sum(in_degree0 > Ne) > 0 or sum(out_degree0 > Ne) > 0 or abs(coef - self.r)/self.r > 0.1:
            in_degree0, out_degree0 = hybrid_degree.hybrid_degree(Ne, degree_mean_ee, degree_std, r, hybrid)
            coef = np.corrcoef(in_degree0,out_degree0)[1,0]
        print('indegree-outdegree correlation: %.4f' %coef)
        
        tic = time.perf_counter()
        i_ee, j_ee, dist_ij_ee, ccoef_list, lattice_ext, glitch = generate_connection.connection(in_degree0, out_degree0, width, tau_p_ee, cn_scale_wire, iter_num, record_cc)
        print('Generating ee_connectivity costs: %.4fs.' %(time.perf_counter() - tic))
        
        A_ee = csr_matrix((np.ones(i_ee.shape, dtype=int), (i_ee, j_ee)),shape=(Ne,Ne))
    
        in_degree_ee = np.sum(A_ee, axis = 0).A.reshape(Ne) 
        #out_degree_ee = np.sum(A_ee, axis = 1).A.reshape(Ne) 
                
        del A_ee
        
        #import inverse_pool
        J_scale = in_degree_ee**0.5
        #mean_J_ee = 4*10**-3 # usiemens
        #sigma_J_ee = 1.9*10**-3 # usiemens
        if self.w_ee_dist == 'lognormal':
            pool_generator = lambda N: g_pool_generator.g_pool_from_g(N, mean_J_ee, sigma_J_ee, 'mu_sigma_log')
            print('Distribution of E-E weight: lognormal')
        if self.w_ee_dist == 'normal':
            pool_generator = lambda N: g_pool_generator.g_pool_from_g_normal(N, mean_J_ee, sigma_J_ee)
            print('Distribution of E-E weight: normal')
        
        tic = time.perf_counter()
        weight_J_ee_tmp = inverse_pool.inverse_pool(in_degree_ee, J_scale, pool_generator) # tseg, tsb
        #weight_J_ee_tmp = inverse_pool_bk.inverse_pool(in_degree_ee, J_scale, pool_generator)
        print('inverse-pooling costs: %.4fs.' %(time.perf_counter() - tic))
        
        tic = time.perf_counter()
        weight_J_ee_tmp = shuffle_weight_common_neighbour.shuffle_weight(weight_J_ee_tmp, i_ee, j_ee, Ne, Ne, cn_scale_weight)
        print('shuffling weight costs: %.4fs.'%(time.perf_counter() - tic))
        
        
        weight_J_ee = np.zeros(j_ee.shape)
        for ind in range(Ne):
            weight_J_ee[j_ee==ind] = weight_J_ee_tmp[ind]
        del weight_J_ee_tmp
                
        return i_ee, j_ee, weight_J_ee, lattice_ext, dist_ij_ee
    
    
    def i_e(self, Ne, Ni, p_ie, i_ee, j_ee, width, weight_J_ee, ie_ratio, lattice_ext, tau_p_ie):
        
        #lattice_inh = quasi_lattice.lattice(width, Ni, centre=[0,0])
        lattice_inh = coordination.makelattice(int(np.sqrt(Ni)), width, centre=[0,0])
        degree_mean_ie = Ne * p_ie
        out_degree_ie = np.random.poisson(lam = degree_mean_ie, size = Ni)
        out_degree_ie = self.check_if_exceed_max(out_degree_ie, degree_mean_ie, Ne)
        
        i_ie, j_ie, dist_ij_ie = connect_2lattice.fix_outdegree(lattice_inh, lattice_ext, out_degree_ie, tau_p_ie, width, src_equal_trg = False)
        
        A_ie = csr_matrix((np.ones(i_ie.shape, dtype=int), (i_ie, j_ie)),shape=(Ni,Ne))
        
        in_degree_ie = np.sum(A_ie, axis = 0).A.reshape(Ne) 
        #out_degree_ie = np.sum(A_ie, axis = 1).A.reshape(Ni)
        
        del A_ie
        
        weight_J_ie = np.zeros(j_ie.shape)
        #weight_J_ee = weight_J_ee/nsiemens
        W_ee = csr_matrix((weight_J_ee,(i_ee,j_ee)))
        W_in_ee = np.sum(W_ee, axis = 0).A.reshape(Ne)
        del W_ee
        
        for i in range(Ne):
            mean_tmp = W_in_ee[i]*ie_ratio/in_degree_ie[i]
            sigma_tmp = mean_tmp*0.25
            weight_J_ie[j_ie==i] = abs(np.random.normal(loc=mean_tmp, scale=sigma_tmp, size=in_degree_ie[i]))
        
        del W_in_ee
        
        return i_ie, j_ie, weight_J_ie, lattice_inh, dist_ij_ie
    
    def e_i_and_i_e(self, Ne, Ni, p_ei, p_ii, lattice_ext, lattice_inh, tau_p_ei, tau_p_ii, width):
        
        degree_mean_ei = Ni*p_ei
        degree_mean_ii = Ni*p_ii
        
        out_degree_ei = np.random.poisson(lam = degree_mean_ei, size = Ne)
        out_degree_ei = self.check_if_exceed_max(out_degree_ei, degree_mean_ei, Ni)
        
        out_degree_ii = np.random.poisson(lam = degree_mean_ii, size = Ni)
        out_degree_ii = self.check_if_exceed_max(out_degree_ii, degree_mean_ii, Ni-1)
        
        i_ei, j_ei, dist_ij_ei = connect_2lattice.fix_outdegree(lattice_ext, lattice_inh, out_degree_ei, tau_p_ei, width, src_equal_trg = False)
        i_ii, j_ii, dist_ij_ii = connect_2lattice.fix_outdegree(lattice_inh, lattice_inh, out_degree_ii, tau_p_ii, width, src_equal_trg = True, self_cnt = False)
        
        return i_ei, j_ei, dist_ij_ei, i_ii, j_ii, dist_ij_ii
    
    def check_if_exceed_max(self, degree, mean, maximum):
        while sum(degree > maximum) > 0:
            degree[degree > maximum] = np.random.poisson(lam = mean, size = sum(degree > maximum))
        
        return degree
    
    def change_ie(self, new_ie_ratio):
        
        self.ie_ratio = new_ie_ratio
        A_ie = csr_matrix((np.ones(self.i_ie.shape, dtype=int), (self.i_ie, self.j_ie)),shape=(self.Ni,self.Ne))
        
        in_degree_ie = np.sum(A_ie, axis = 0).A.reshape(self.Ne) 
        #out_degree_ie = np.sum(A_ie, axis = 1).A.reshape(Ni)
        
        del A_ie
        
        self.w_ie = np.zeros(self.j_ie.shape)
        #weight_J_ee = weight_J_ee/nsiemens
        W_ee = csr_matrix((self.w_ee,(self.i_ee,self.j_ee)))
        W_in_ee = np.sum(W_ee, axis = 0).A.reshape(self.Ne)
        del W_ee
        
        for i in range(self.Ne):
            mean_tmp = W_in_ee[i]*new_ie_ratio/in_degree_ie[i]
            sigma_tmp = mean_tmp*0.25
            self.w_ie[self.j_ie==i] = abs(np.random.normal(loc=mean_tmp, scale=sigma_tmp, size=in_degree_ie[i]))
        
        del W_in_ee
        
        
#%%
'''
def test(self, Ni, Ne):
    
    def plus(a,b):
        return a+b  
    
    
    
    c=plus(Ni, Ne)
    d=plus(Ni, Ne)
    
    return cplus
'''
#%%
#c1 = get_ijw()
#c1.generate()
#
#
#c2 = get_ijw()
#c2.generate()
#
##%%
#
#c3 = get_ijw()
#c3.i_ee, c3.j_ee, c3.w_ee, c3.lattice_ext = c2.i_ee, c2.j_ee, c2.w_ee, c2.lattice_ext
#
#c3.generate_ie()
##%%
#c1 = get_ijw(dict(Ne=64*64,Ni=1010),{'p_ee':0.15},cv=0.3)
##%%
#m=10
#dict(m=10, n=5)











    
