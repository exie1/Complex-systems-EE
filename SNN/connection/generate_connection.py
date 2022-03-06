#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:36:51 2019

@author: shni2598
"""
import numpy as np
import numpy.ma as ma
from connection import coordination
#from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
#from numba import jit
#import pdb  

#nan
#isnan
def connection(degree_in, degree_out, width, tau_d, cn_scale_wire, iter_num, record_cc=False):
#%%  
#    iter_num = 5
#    cn_scale_wire = 2
#    #tau_d = 8
#    width = 39
#    tau_d = width/(int(N**0.5) - 1)*8
#    degree_in=in_degree0; degree_out=out_degree0
#    record_cc=True
    #hw = width/2
    N = len(degree_in)
    syn_num = np.sum(degree_in.astype(int)) # total number of synapses
    
    lattice = coordination.makelattice(int(N**0.5), width, [0, 0])
    
    if cn_scale_wire < 1:
        raise Exception('Error: cn_scale_wires should not be less than 1!')
    cn=0
    for itr in range(iter_num):
        #
        #itr = 1; 
        glitch = 0 # record how many source neurons have choosen target neurons which have already exceeded their maximum input-synapses number (in-degree)
        #
        i = np.zeros(syn_num, dtype=int) #i = np.array([],dtype=int)
        j = np.zeros(syn_num, dtype=int) #j = np.array([],dtype=int)
        dist_ij = np.zeros(syn_num, dtype=float)#dist_ij = np.array([])
        pre_ind = 0
        degree_in_left = degree_in
        degree_in_left = degree_in_left.astype(int) # make sure it is type int, for numerical accuracy during operation
        degree_in_left = ma.array(degree_in_left, mask=np.zeros(len(degree_in_left),dtype=bool)) # converted to mask array
        
        if itr > 0 and cn_scale_wire > 1:
            cn_min = cn.min(); cn_max = cn.max() # pre-define cn_min/max before iteration below to speed up computation
        if itr == 0 or cn_scale_wire ==1:
            cn_factor = np.ones(degree_in.shape) # pre-define cn_factor before iteration below to speed up computation
        
#        np.random.seed(2)        
        randN = np.random.permutation(range(N))
#        randN = np.arange(N)
        neuron_itr = 1
        n_ignore_indegree_limit = N - 40 #define when the source neuron can select target neurons who may have reached their in-degree limit, i.e., ignore indegree
        for neuron in randN:
            #neuron=1
            # distance factor
            neuron = [neuron]
            post_dist = coordination.lattice_dist(lattice, width, neuron)
            neuron = neuron[0]
            dist_factor = np.exp(-post_dist/tau_d)
            dist_factor[neuron] = 0
            
            #pdb.set_trace()###
            # cn_factor                  
#            if itr == 0 or cn_scale_wire == 1:
#                cn_factor = np.ones(dist_factor.shape)
            if itr > 0 and cn_scale_wire > 1:
                cn_factor = 1 + ((cn[neuron,:].toarray() - cn_min))/(cn_max - cn_min)*(cn_scale_wire - 1)
                cn_factor = cn_factor.reshape(N)
                #cn_factor = cn_factor.reshape(cn_factor.shape[0])
            # degree factor
            degree_in_factor = degree_in_left
            degree_in_factor = degree_in_factor.astype(float)
            degree_in_factor[degree_in_factor<=4] = degree_in_factor[degree_in_factor<=4]**6/(4**6)
            if neuron_itr > n_ignore_indegree_limit: # (<0.005*N)at the end of iteration, available neurons to be selected are not much.
                degree_in_factor = ma.array(np.ones(N, dtype=float),mask=np.zeros(N))
                degree_in_factor.mask[neuron] = True # avoid 'divided by zero' warning. don't know why masked array can avoid this warning.
                #pdb.set_trace()
            #if neuron_itr > 3500: pdb.set_trace() ###
             
            joint_factor = dist_factor*cn_factor*degree_in_factor
            joint_factor = joint_factor/(np.nansum(joint_factor**2))**0.5
            #pdb.set_trace()
            #np.random.seed(2) 
            choose_j = np.argsort(np.random.rand(N)/joint_factor)[:int(degree_out[neuron])]
            
#            pdb.set_trace() ###
            if np.sum(degree_in_left.mask[choose_j]) > 0:
                glitch += 1
            
            #degree_in_left = np.round(degree_in_left).astype(int)
            degree_in_left[choose_j] -= 1
            #degree_in_left = degree_in_left.astype(float)
            degree_in_left.mask[degree_in_left==0] = True
#            '''
#            i = np.concatenate((i, neuron*np.ones(len(choose_j), dtype=int)))
#            j = np.concatenate((j, choose_j.astype(int)))
#            
#            dist_ij = np.concatenate((dist_ij, post_dist[choose_j]))
#            '''
            choose_len = len(choose_j)
            i[pre_ind:pre_ind+choose_len] = neuron*np.ones(choose_len, dtype=int)
            j[pre_ind:pre_ind+choose_len] = choose_j.astype(int)
            dist_ij[pre_ind:pre_ind+choose_len] = post_dist[choose_j]
            
            pre_ind += choose_len
            
            if neuron_itr%500 == 0:
                print('iter: %d, neuron: %d' %(itr, neuron_itr))
            neuron_itr += 1
        
        if (iter_num > 1 and cn_scale_wire > 1 and itr < iter_num - 1) or record_cc ==True:
            A = csr_matrix((np.ones(i.shape), (i, j)),shape=(N,N))
        
        if iter_num > 1 and cn_scale_wire > 1 and itr < iter_num - 1:            
            cn = np.dot(A.T, A) # note that cn is a symmetric matrix 
            cn = cn.tolil() # optional, for computational efficiency
            cn[np.eye(cn.shape[0],dtype=bool)] = 0
            cn = cn.tocsr()
            #cn = cn.toarray()
        
        if record_cc == True:
            if itr == 0:
                ccoef_list = [None]*iter_num
            hf_sd_n = 30    
            # if half of the number of neurons along each side exceeds 32, select subset of network to compute clustering-coef 
            # in order to save time.
            # However, the clustering-coef of sub-network may be very different from that of whole network. I don't know why.
            # Update: Use the square rather than circle as sample area, this may decrease the error of clustering-coef.
            if (N**0.5 - 1)/2 > hf_sd_n:
#                sample_ind = (lattice[:,0]**2 + lattice[:,1]**2) <= ((width/(N**0.5-1))*hf_sd_n)**2 # (width/2)**2
#                ccoef = clustering_coef_wd(A[sample_ind,:][:,sample_ind])
                sample_ind = (abs(lattice[:,0]) < hf_sd_n) & (abs(lattice[:,1]) < hf_sd_n)
                ccoef = clustering_coef_wd(A[sample_ind,:][:,sample_ind])
            else:
                ccoef = clustering_coef_wd(A)
            
            ccoef_list[itr] = ccoef #ccoef_list.append(ccoef)
        else:
            ccoef_list = 0
    
    return i, j, dist_ij, ccoef_list, lattice, glitch
            
#%%
def clustering_coef_wd(W):
    #W = W.A
    #W = A
    A =(W != 0).astype(float) # adjacent matrix
    S = W.power(1/3) + W.T.power(1/3)
    del W
    K = np.sum((A + A.T), axis=0)
    K = np.array(K)
    K = K.reshape(K.shape[1])
    #cyc3 = np.diag(S*S*S)/2
    cyc3 = (S*S*S).diagonal()*0.5
    K[cyc3==0] = np.inf
    #CYC3 = K*(K-1) - 2*np.diag(np.dot(A,A))
    CYC3 = K*(K-1) - 2*(A*A).diagonal()
    del A, K
    ccoef = cyc3/CYC3
    
    return ccoef

    
