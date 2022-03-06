# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 19:20:00 2019

@author: nishe
"""

from connection import g_EPSP_conversion
import numpy as np
#from numba import jit
import pdb

def g_pool_from_potential(N, mean_EPSP, sigma_EPSP, logornorm):
    
    N = int(N)
    EPSP_max = 20 # mV
#    print('Note that log-bin correction is being used during the g_pool_generation')
    
#    mean_EPSP = -0.702
#    sigma_EPSP = 0.9355
    
    if logornorm.lower() == 'mu_sigma_log':
        mean_EPSP_norm = np.log((mean_EPSP**2)/np.sqrt(sigma_EPSP**2+mean_EPSP**2))
        sigma_EPSP_norm = np.sqrt(np.log(sigma_EPSP**2/(mean_EPSP**2)+1))
        
    elif logornorm.lower() == 'mu_sigma_norm':
        mean_EPSP_norm = mean_EPSP - sigma_EPSP**2 #log-bin correction 
        sigma_EPSP_norm = sigma_EPSP
        print('Note that log-bin correction is being used during the g_pool_generation')
    
    else:
        raise TypeError('Missing ''logornorm'' argument. You need to tell if ''mu'' and ''sigma'' specify the whole lognormal(log) or the power(normal) part of lognormal.')
    
#    mean_EPSP_norm = mean_EPSP - sigma_EPSP**2 #log-bin correction 
#    sigma_EPSP_norm = sigma_EPSP
#    pdb.set_trace()
    EPSP = np.random.lognormal(mean_EPSP_norm, sigma_EPSP_norm, N )
    
    while max(EPSP) > EPSP_max:
        EPSP[EPSP>EPSP_max] = np.random.lognormal(mean_EPSP_norm, sigma_EPSP_norm, sum(EPSP>EPSP_max) )
    
    fit_EPSP_to_g = g_EPSP_conversion.g_EPSP_convert()[1]
    #gfit,pfit = g_EPSP_conversion.g_EPSP_convert()[2:4]
    #pfit = g_EPSP_conversion.g_EPSP_convert()[3]
    #fit = fit[1]
    g = fit_EPSP_to_g.predict(EPSP.reshape(-1,1))
    g = g.reshape(N)
    
    return g
#@jit(nopython=False)
def g_pool_from_g(N, mean_g, sigma_g, logornorm):
    
    # g (or J, or w): connection/synapse strength (unit: siemens)
    N = int(N)
    g_max = 38.32*10**-3
    
    if logornorm.lower() == 'mu_sigma_log':
        mean_g_norm = np.log((mean_g**2)/np.sqrt(sigma_g**2+mean_g**2))
        sigma_g_norm = np.sqrt(np.log(sigma_g**2/(mean_g**2)+1))
        
    elif logornorm.lower() == 'mu_sigma_norm':
        mean_g_norm = mean_g
        sigma_g_norm = sigma_g
    
    else:
        raise TypeError('Missing ''logornorm'' argument. You need to tell if ''mu'' and ''sigma'' specify the whole lognormal(log) or the power(normal) part of lognormal.')
    
    g = np.random.lognormal(mean_g_norm, sigma_g_norm, N )

    while np.max(g) > g_max:
        g[g>g_max] = np.random.lognormal(mean_g_norm, sigma_g_norm, np.sum(g>g_max))
        
    g = g.reshape(N)

    return g

def g_pool_from_g_normal(N, mean_g, sigma_g):
    
    N = int(N)
    g = mean_g + sigma_g*np.random.randn(N)
    
    while np.min(g) <= 0:
        g[g<=0] = mean_g + sigma_g*np.random.randn(np.sum(g<=0))
        
    return g.reshape(-1)
            

    
#fit_EPSP_to_g.predict([[0]])
#gg = g_pool_generate(1000,-0.702,0.9355)
    
#import matplotlib.pyplot as plt     
#plt.plot(np.arange(len(rec)),rec)
#plt.plot(gfit,pfit)
#plt.hist(np.log(EPSP),bins=20)
#plt.hist(g,bins=20)
    
#%%
#np.log((4**2)/np.sqrt(1.9**2+4**2))
#Out[9]: 1.284568902135654
#
#np.sqrt(np.log(1.9**2/(4**2)+1))
#Out[10]: 0.4510553380334542    
#from scipy.stats import lognorm
#from scipy.stats import norm
#
#s = 0.4510553380334542
#loc = 1.284568902135654
#g_50 = lognorm.ppf( 0.5, s=1.9, loc = 4 )
#g_98 = lognorm.ppf( 0.98, s=s, loc = loc )
#
#
#fit_p2g = g_EPSP_conversion.g_EPSP_convert()[1]
#fit_g2p, fit_p2g = g_EPSP_conversion.g_EPSP_convert()#[0]
#fit_g2p = g_EPSP_conversion.g_EPSP_convert()[0]
#
#fit_g2p = fit_g2p[0]
#g_2p = fit_p2g.predict([[2]]) # us 2mv
#g_5p = fit_p2g.predict([[5]]) # us 5mv
#
#norm.ppf(0.98, scale=1.9, loc=4)
#
#fit_g2p.predict([[0.005]])




