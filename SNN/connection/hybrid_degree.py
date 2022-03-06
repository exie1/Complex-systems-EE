# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:23:02 2019

@author: nishe
"""
# generate the input and output degree for each neuron, the degree is a hybrid of
# poisson and lognormal distribution
# hybrid: the hybrid degree of poisson and lognormal distribution
from connection import mypoiss
from connection import mylognorm
import numpy as np

def hybrid_degree(N, deg_mean, deg_std, r, hybrid):
    mismatch_tol = deg_mean/2
    mismatch = mismatch_tol + 1
    
    while abs(mismatch) > mismatch_tol:
        deg = hybrid_poiss_logn(deg_mean, deg_std, r, N, hybrid); # deg[0,:]: input deg; deg[1,:]: output deg
        mismatch = sum(deg[0,:]) - sum(deg[1,:])
    
    adjust_ind = np.random.permutation(range(N))[:int(abs(mismatch))] 
    if mismatch != 0:     # make the input and output degree equal
        if mismatch > 0:
            deg[1,adjust_ind] += 1
        else:
            deg[0,adjust_ind] += 1
    
    in_degree = deg[0,:].astype(int)
    out_degree = deg[1,:].astype(int)
    
    return in_degree, out_degree
                    

def hybrid_poiss_logn(deg_mean, deg_std, r, N, hybrid):
    
    deg_logn = np.ceil(mylognorm.logn([deg_mean, deg_mean], deg_std, [[1,r],[r,1]], N, 'mu_sigma_log'))
    deg_poiss = mypoiss.poiss([deg_mean, deg_mean], r, N)
    
    deg = deg_poiss
    # logn_ind = np.random.permutation(range(N))[:round(N*hybrid)]
    
    if hybrid > 0:
        logn_ind = np.random.permutation(range(N))[:round(N*hybrid)]
        deg[:,logn_ind] = deg_logn[:,logn_ind]
        
    return deg
    

