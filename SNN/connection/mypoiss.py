#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:27:46 2019

@author: shni2598
"""
#%%
from scipy.stats import poisson
from scipy.stats import norm
import numpy as np
#%%
# generate 2 arrays of correlated poisson variables
def poiss(lam, r, N):    
#    lam = [5,5]
#    r=0.2; N=100
    mean_norm = [0, 0]
    cov = [[1, r], [r, 1]] 
    
    err = 1; err_max = 0.1    
    max_iteration = 100
    iteration = 0
    
    while err > err_max:
        iteration += 1
        if iteration > max_iteration:
            raise Exception('Error: Cannot converge.')
        
        corr_normal = np.random.multivariate_normal(mean_norm, cov, N).T # 2D correlated normal
        
        corr_unif = norm.cdf(corr_normal) # correlated uniform
        
        pois = np.zeros(corr_unif.shape)
        pois[0] = poisson.ppf(corr_unif[0], lam[0])
        pois[1] = poisson.ppf(corr_unif[1], lam[1])
        if r > 0.05:
            err = abs((np.corrcoef(pois[0],pois[1])[0,1] - r)/r)
        else:
            err = abs(np.corrcoef(pois[0],pois[1])[0,1] - r)
    
    return pois
    #np.mean(pois[0])
    #np.var(pois[0])
    #poisson.ppf(0.98,5)
