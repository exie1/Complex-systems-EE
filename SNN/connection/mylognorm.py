#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:08:41 2019

@author: shni2598
"""

import numpy as np
#%%
# Generate 2 arrays of correlated lognormal variables 
def logn(mu, sigma, corrmat, N, logornorm):   
#    mu=[1,1]; sigma=[2,2]
#    corrmat=[[1,0.2],[0.2,1]]
#    logornorm = 'mu_sigma_log'
    mu = np.array(mu)
    sigma = np.array(sigma)
    corrmat = np.array(corrmat)
    if logornorm.lower() == 'mu_sigma_log':
        mu_norm = np.log((mu**2)/np.sqrt(sigma**2+mu**2))
        sigma_norm = np.sqrt(np.log(sigma**2/(mu**2)+1))
    
    elif logornorm.lower() == 'mu_sigma_norm':
        mu_norm = mu
        sigma_norm = sigma
    
    else:
        raise TypeError('Missing ''logornorm'' argument. You need to tell if ''mu'' and ''sigma'' specify the whole lognormal(log) or the power(normal) part of lognormal.')
#        print('Error: You need to tell if ''mu'' and ''sigma'' specify the whole lognormal(log) or the power(normal) part of lognormal')
#        br
    cov = np.zeros([2,2])
    
    cov[0,0] = sigma_norm[0]**2
    cov[1,1] = sigma_norm[1]**2
    r = corrmat[0,1]
    cov[0,1] = np.log(r*np.sqrt((np.exp(sigma_norm[0]**2)-1)*(np.exp(sigma_norm[1]**2)-1))+1)
    cov[1,0] = cov[0,1]
    
    err = 1; err_max = 0.1
    max_iteration = 100
    iteration = 0
    
    while err > err_max:
        iteration += 1
        
        if iteration > max_iteration:
            raise Exception('Error: Cannot converge.')
        
        lognorm = np.exp(np.random.multivariate_normal(mu_norm, cov, N).T)
        err = abs((np.corrcoef(lognorm[0],lognorm[1])[0,1] - r)/r)
        
    return lognorm
#%%
#lognorm=logn([1,1],[2,2],[[1, 0.2],[0.2,1]],1000,'mu_sigma_log')
#
#np.corrcoef(lognorm[0],lognorm[1])
#np.mean(lognorm[0])
#np.var(lognorm[0])
#
#radn = np.random.multivariate_normal(mu_norm, cov, 1000).T
#radlogn = np.exp(radn[0])
#
#
#import matplotlib.pyplot as plt
#plt.hist(np.log(lognorm[0]),bins=50)
#plt.hist(lognorm[0],bins=50)
#
#
##plt.hist(radn[0],bins=50)#,range=[0,5])
##plt.figure()
#plt.hist(radlogn,bins=50)
#plt.hist(np.log(radlogn),bins=50)#,range=[0,5])
#
#a=testerror.test(3,2)
