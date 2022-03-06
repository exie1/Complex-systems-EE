#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:45:31 2019

@author: shni2598
"""

import numpy as np
from numpy.random import permutation
#import pdb  
#import time

def inverse_pool(J_num, J_scale, pool_generator):
    
    #t1 = []
    #t2 = []
    
    max_J_pool_num = 10
    err_max = 0.1
    max_sb_round = 4
    max_err_round = 50
    pool_length_extra = 0.01
    
    pool_length = round(np.sum(J_num)*(1+pool_length_extra))
    J_sum_tot = np.sum(pool_generator(np.sum(J_num)))
    J_sum = J_scale/np.sum(J_scale)*J_sum_tot
    N = len(J_num) # number of neurons
    
    J_num_sort_ind = np.argsort(J_num)
    #N=9
    azby = np.zeros(N)
    azby[0::2] = np.arange(np.ceil(N/2))
    azby[1::2] = np.arange(np.ceil(N/2),N)[::-1]
    azby = azby.astype(int)
    
    J_num_azby = J_num[J_num_sort_ind[azby]]
    i_left_azby = J_num_sort_ind[azby]
    J_sum_azby = J_sum[J_num_sort_ind[azby]]
    
    J = [None]*N
    #J_std = np.zeros(N)
    J_pool_num = 0
    J_pool_left = np.array([])
    err_acc = 0
    
    while np.sum(i_left_azby==-1) < N and J_pool_num < max_J_pool_num:
        J_pool_num += 1
        print('Generate J: J_pool number = %d.\n' %J_pool_num)
        
        J_pool = pool_generator(pool_length) # function holder
        J_pool_left = np.concatenate((J_pool_left, J_pool))
        J_pool_left = np.sort(J_pool_left)
        
        # quick check for solvability
        solvable = np.ones(N,dtype=bool)
        
        for i_check in range(N):
            
            if np.sum(J_pool_left[0:J_num[i_check]]) > J_sum[i_check] or np.sum(J_pool_left[-J_num[i_check]::]) < J_sum[i_check]:
                solvable[i_check] = False
        
        N_unsolvable = np.sum(np.logical_not(solvable))
        
        if N_unsolvable > 0 and N_unsolvable < 0.05*N:
            print('There are %d (J_num, J_sum) pair cannot be solved\n' %N_unsolvable)
            print('Trying another pool...\n')
            J_pool_left = np.array([])
            continue
        
        elif N_unsolvable >= 0.05*N:
            print('There are %d (J_num, J_sum) pair cannot be solved\n' %N_unsolvable)
            print('Inverse pool abandoned. Please adjust the input arguments!\n')
            break
    
        else:
            for ind in range (N):
                
                ii = i_left_azby[ind] #  the ii^th neuron
                if ind%500 == 0: print('neuron:%d' %ind)
                
                if ii != -1:
                    sb_round = 0
                    J_sum_tmp = J_sum_azby[ind]
                    J_num_tmp = J_num_azby[ind]
                    
                    while sb_round < max_sb_round:  
                        sb_round += 1
                        #toc = time.perf_counter()
                        N_s_tot, N_b_tot, N_s, N_b = find_sb_sep(J_pool_left, J_sum_tmp, J_num_tmp)
                        #t2.append(time.perf_counter()-toc)
                        J_found = False
                        
                        if not(np.isnan(N_s_tot)):
                            err_round = 0
                            s_samp = 0
                            b_samp = 0
                            #tic = time.perf_counter()
                            while err_round < max_err_round:
                                
                                #err_round += 1
                                
                                #pdb.set_trace()
                                #sub_s = np.random.choice(N_s_tot, N_s, replace=False)
                                #sub_b = np.random.choice(np.arange(N_s_tot, N_s_tot+N_b_tot), N_b, replace=False)
                                if err_round == 0 or (s_samp+1)*N_s > N_s_tot:
                                    s_perm = permutation(N_s_tot); s_samp = 0
                                if err_round == 0 or (b_samp+1)*N_b > N_b_tot:
                                    b_perm = permutation(N_b_tot); b_samp = 0
                                
                                sub_s = s_perm[s_samp*N_s:(s_samp+1)*N_s]; s_samp += 1
                                sub_b = N_s_tot + b_perm[b_samp*N_b:(b_samp+1)*N_b]; b_samp += 1
                                #sub_s = permutation(N_s_tot)[:N_s]
                                #sub_b = N_s_tot+ permutation(N_b_tot)[:N_b]
                                                                
                                J_selected_tmp = J_pool_left[permutation(np.concatenate((sub_s,sub_b)))]
                                
                                err = np.abs((np.sum(J_selected_tmp) - J_sum_tmp)/J_sum_tmp)
                                err_sign = np.sign(np.sum(J_selected_tmp) - J_sum_tmp)
                                err_sign_match = err_sign == np.sign(err_acc)
                                
                                if err <= err_max and not(err_sign_match):
                                    
                                    #t1.append(time.perf_counter() - tic)
                                    i_left_azby[ind] = -1
                                    J[ii] = J_selected_tmp
                                    #J_std[ii] = np.std(J_selected_tmp) # record standard deviation
                                    J_pool_left = np.delete(J_pool_left, np.concatenate((sub_s,sub_b)))
                                    err_acc = err_acc + np.sum(J_selected_tmp) - J_sum_tmp
                                    sb_round = max_sb_round # help to break out of the outer while loop
                                    J_found = True
                                    #t1.append(time.perf_counter() - tic)
                                    break
                                
                                err_round += 1
                        
                        if not(J_found) and sb_round == max_sb_round:
#                        if np.isnan(N_s_tot) and sb_round == max_sb_round:
                            print('Neuron:%d is unsolvable.Try it in the next pool.\n' %ii)
                            
        print('%d out of %d reverse-pooled.\n' %(np.sum(i_left_azby==-1), N))
    
    if sum(i_left_azby==-1) < N:
        J = []
        print('Inverse-pooling abandoned. Adjust the input arguments!\n')
    
    return J#, t1 ,t2
                                
                        
                        

def find_sb_sep(J_pool_left, J_sum, J_num):
    sb_sep_found = False
    while_count = 0
    max_while_count = 2*np.log2(len(J_pool_left))
    pool_length = len(J_pool_left)
    sep_lower_limit = 1;
    sep_upper_limit = len(J_pool_left)
    
    while while_count < max_while_count and sep_upper_limit - sep_lower_limit > 0 and pool_length >= J_num:
        while_count += 1
        adjust_limits = 0
        sb_sep = np.random.randint(sep_lower_limit,sep_upper_limit)
        mean_s = np.mean(J_pool_left[:sb_sep])
        mean_b = np.mean(J_pool_left[sb_sep:])
        
        if mean_s > J_sum/J_num:
            sep_upper_limit = sb_sep
        elif mean_b < J_sum/J_num:
            sep_lower_limit = sb_sep + 1
        else:
            N_s_tot = sb_sep
            N_b_tot = pool_length - sb_sep
            A = np.array([[mean_s,mean_b],[1,1]])
            B = np.array([J_sum,J_num])
            N_sb = np.linalg.solve(A,B)
            N_s = N_sb[0]
            N_b = N_sb[1]
            # adjust the searching boundary
            if N_s > N_s_tot:
                sep_lower_limit = sb_sep + 1
                adjust_limits = 1
            if N_b > N_b_tot:
                sep_upper_limit = sb_sep
                adjust_limits = adjust_limits + 1
            if adjust_limits == 0:
                #pdb.set_trace()
                if N_s < N_b:
                    N_s =int(np.round(N_s))
                    N_b = J_num - N_s
                else:
                    N_b = int(np.round(N_b))
                    N_s = J_num - N_b
                sb_sep_found = True
                #N_s_tot = int()
                break
    
    if not(sb_sep_found):
        N_s_tot = np.nan
        N_b_tot = np.nan
        N_s = np.nan
        N_b = np.nan
    
    return N_s_tot, N_b_tot, N_s, N_b
                
                
