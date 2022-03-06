#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:20:41 2019

@author: shni2598
"""
import numpy as np
#from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
import pdb
def shuffle_weight(w, i, j, i_len, j_len, cn_scale):
    
    if cn_scale > 1:
          
        A = csr_matrix((np.ones(i.shape), (i,j)), shape=(i_len, j_len))
        
        cn = A.T * A
        cn = cn.tolil() # optional, for computational efficiency
        cn[np.eye(cn.shape[0],dtype=bool)] = 0
        cn = cn.tocsr()
        
        N = len(w) #max(j)
        #pdb.set_trace()
        w_max =  max([max(item) for item in w]) #max(w)
        w_min =  min([min(item) for item in w]) #min(w) 
        cn_max = cn.max()
        cn_min = cn.min()
        
        for j_ind in range(N):
            
            cn_tmp = cn[i[j==j_ind], j_ind].A.reshape(-1)
            w_tmp = w[j_ind].copy()
            w_shuffle = np.zeros(w_tmp.shape)
            
            cw_factor = 1+ (cn_tmp - cn_min)/(cn_max - cn_min)*(w_tmp - w_min)/(w_max - w_min)*(cn_scale - 1)
            #pdb.set_trace()
            sort_ind = np.argsort(np.random.rand(cw_factor.shape[0])/cw_factor)
            
            w_tmp[::-1].sort()
            w_shuffle[sort_ind] = w_tmp
            
            w[j_ind] = w_shuffle
        
        return w
        
    elif cn_scale == 1:
        
        return w
    
    else :
        raise Exception('Error: cn_scale should be no less than 1.')
        
    

