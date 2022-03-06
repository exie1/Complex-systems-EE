# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:20:08 2019

@author: nishe
"""

from connection import post_analysis
import pickle
#from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Qt5Agg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#from sklearn.linear_model import LinearRegression

import brian2.numpy_ as np
from brian2.only import * 

#%%

## spike: an object containing the spike time and the index of corresponding neruon. 
## It has attributes i and t; i is the index of neuron generating spike, t is the time of spike.
## N_row: Number of neurons
## N_col: total timesteps of simulation 
## cv_window: length of window to calculate cv (unit: time)
## aval_window: length of window to detect avalanche (unit: time)
## dt: simulation timestep (unit: time)
## discard_t: the length of initial simulation period to ignore in order to exclude transient period.
## merge_cv_num: number of cv_window to merge when calculate the avalanche distribution.
## sys_arg: PBS array_job id
## save_fig: if save the figure

def avalanche(spike, N_row, N_col, cv_window, rate_window, aval_window, dt, discard_t, merge_cv_num,
              sys_arg, save_fig, show_fig=False):
    
    #cv_window=0.5*second, rate_window=50*ms, aval_window=0.1*ms, discard_t=200*ms
    anly = post_analysis.analysis()
    #aval_st,
    aval_st, cv = anly.detect_avalanche_window(spike, row=N_row, col=N_col, cv_window=cv_window, 
                                               rate_window=rate_window, 
                                               aval_window=aval_window, dt=dt, discard_t=discard_t)
            
    #merge_cv_num = 3 
    aval_group_e = anly.sortandgroup_cv(cv, aval_st, merge_cv_num)    

    for num_grp in range(len(aval_group_e)):
    
        cv_e_sort = np.sort(cv)
        cv_grp = np.mean(cv_e_sort[num_grp*merge_cv_num:(num_grp+1)*merge_cv_num])
        
        freqt_e, abinst_e, bt1, coef_et, figt_e, axt_e = anly.visualize_aval_distribution(aval_group_e[num_grp][0,:], drange = None, returnfig = True, nbins = 50)
        axt_e.set_xlabel('log(t)')
        if save_fig: figt_e.savefig('t%s_%s.png'%(num_grp,sys_arg))
        if not(show_fig): plt.close(figt_e)
        
        freqs_e, abinss_e, bs1, coef_es, figs_e, axs_e = anly.visualize_aval_distribution(aval_group_e[num_grp][1,:], drange = None, returnfig = True, nbins = 50)
        axs_e.set_xlabel('log(s)')
        if save_fig: figs_e.savefig('s%s_%s.png'%(num_grp,sys_arg))
        if not(show_fig): plt.close(figs_e)
        
        figts_e, axts_e, coef_ets = anly.visualize_t_size_distribution(aval_group_e[num_grp][0,:], aval_group_e[num_grp][1,:], returnfig = True)
        
        if save_fig: figts_e.savefig('ts%s_%s.png'%(num_grp,sys_arg))
        if not(show_fig): plt.close(figts_e)
        
        with open('cv_scaling_%s.txt'%sys_arg, 'a') as file:
            file.write('%s,%s,%s,%s,%s\n' %(cv_grp, coef_et, coef_es, (coef_et+1)/(coef_es+1), coef_ets))
    





