# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:10:25 2019

@author: nishe
"""
import numpy as np
from sklearn.linear_model import LinearRegression
#%%
def g_EPSP_convert():
    
    dt = 0.1 # ms
    Cm = 0.25 # nF
    V_lk = -70.0 # leak reversal mV
    g_lk = 0.0167 # muS
    V_rev = 0 # for AMPA
    
    tau_r = 1.0 # ms
    tau_d = 5 # ms
    #V_th = -50 # threshold
    
    g_min = 0.001
    g_max = 0.01
    
    g = np.linspace(g_min,g_max,10)
    EPSP = np.zeros(g.shape)
    #%%
    for i in range (len(g)):
        #g[i]
        #i = 0
        #rec = []
        rise_left = int(tau_r/dt)
        peak = False
        s = 0
        V = V_lk
        Vold = V_lk
        #ds = 0
        while not(peak):        
            if rise_left >0:
                ds = (-s/tau_d + (1/tau_r)*(1-s))*dt
                rise_left -= 1
            else:
                ds = (-s/tau_d)*dt
            s += ds
            #s_rec[i] = s
            dv = ((1/Cm)*((-1)*g_lk*(V - V_lk) + (-1)*g[i]*s*(V - V_rev)))*dt
            V += dv
            if V < Vold:
                peak = True
                EPSP[i] = Vold - V_lk
            Vold = V
            #rec.append(Vold)
    #%%
    #linear_regressor = LinearRegression(fit_intercept=False) # force the intercept to be 0
    g_to_EPSP = LinearRegression(fit_intercept=False)
    g_to_EPSP = g_to_EPSP.fit(g.reshape(-1,1),EPSP.reshape(-1,1))
    EPSP_to_g = LinearRegression(fit_intercept=False)
    EPSP_to_g = EPSP_to_g.fit(EPSP.reshape(-1,1),g.reshape(-1,1))
    
    return g_to_EPSP, EPSP_to_g#, g, EPSP
    
#    import matplotlib.pyplot as plt     
#    plt.plot(np.arange(len(rec)),rec)
#    plt.plot(g,EPSP)
